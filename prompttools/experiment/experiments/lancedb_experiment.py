# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import itertools
import warnings

import pandas as pd
from typing import Callable, Optional

try:
    import lancedb
    from lancedb.embeddings import with_embeddings

except ImportError:
    lancedb = None

import logging
from time import perf_counter
from .experiment import Experiment
from ._utils import _get_dynamic_columns

VALID_TASKS = [""]


def query_builder(
    table: "lancedb.Table",
    embed_fn: Callable,
    text: str,
    metric: str = "cosine",
    limit: int = 3,
    filter: str = None,
    nprobes: int = None,
    refine_factor: int = None,
):
    if nprobes is not None or refine_factor is not None:
        warnings.warn(
            "`nprobes` and `refine_factor` are not used by the default `query_builder`. "
            "Feel free to open an issue to request adding support for them."
        )
    query = table.search(embed_fn(text)[0]).metric(metric)
    if filter:
        query = query.where(filter)
    return query.limit(limit).to_df()


class LanceDBExperiment(Experiment):
    r"""
    Perform an experiment with ``LanceDB`` to test different embedding functions or retrieval arguments.
    You can query from an existing table, or create a new one (and insert documents into it) during
    the experiment.

    Args:
        uri (str): LanceDB uri to interact with your database. Default is "lancedb"
        table_name (str): the table that you will get or create. Default is "table"
        use_existing_table (bool): determines whether to create a new collection or use
            an existing one
        embedding_fns (list[Callable]): embedding functions to test in the experiment
            by default only uses the default one in LanceDB
        query_args (dict[str, list]): parameters used to query the table
            Each value is expected to be a list to create all possible combinations
        data (Optional[list[dict]]): documents or embeddings that will be added to
            the newly created table
        text_col_name (str): name of the text column in the table. Default is "text"
        clean_up (bool): determines whether to drop the table after the experiment ends
    """

    def __init__(
        self,
        embedding_fns: dict[str, Callable],
        query_args: dict[str, list],
        uri: str = "lancedb",
        table_name: str = "table",
        use_existing_table: bool = False,
        data: Optional[list[dict]] = None,
        text_col_name: str = "text",
        clean_up: bool = False,
    ):
        if lancedb is None:
            raise ModuleNotFoundError(
                "Package `lancedb` is required to be installed to use this experiment."
                "Please use `pip install lancedb` to install the package"
            )
        self.table_name = table_name
        self.use_existing_table = use_existing_table
        self.embedding_fns = embedding_fns
        if use_existing_table and data:
            raise RuntimeError("You can either use an existing collection or create a new one during the experiment.")
        if not use_existing_table and data is None:
            raise RuntimeError("If you choose to create a new collection, you must also add to it.")
        self.data = data if data is not None else []
        self.argument_combos: list[dict] = []
        self.text_col_name = text_col_name
        self.db = lancedb.connect(uri)
        self.completion_fn = self.lancedb_completion_fn
        self.query_args = query_args
        self.clean_up = clean_up
        super().__init__()

    def prepare(self):
        for combo in itertools.product(*self.query_args.values()):
            self.argument_combos.append(dict(zip(self.query_args.keys(), combo)))

    def run(self, runs: int = 1):
        input_args = []  # This will be used to construct DataFrame table
        results = []
        latencies = []
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()

        for emb_fn_name, emb_fn in self.embedding_fns.items():
            if self.use_existing_table:  # Use existing table
                table = self.db.open_table(self.table_name)
                if not table:
                    raise RuntimeError(f"Table {self.table_name} does not exist.")
            else:  # Create table and insert data
                data = with_embeddings(emb_fn, self.data, self.text_col_name)
                table = self.db.create_table(self.table_name, data, mode="overwrite")

            # Query from table
            for query_arg_dict in self.argument_combos:
                query_args = query_arg_dict.copy()
                for _ in range(runs):
                    start = perf_counter()
                    results.append(self.lancedb_completion_fn(table=table, embedding_fn=emb_fn, **query_args))
                    latencies.append(perf_counter() - start)
                    query_args["emb_fn"] = emb_fn_name  # Saving for visualization
                    input_args.append(query_args)

            # Clean up
            if self.clean_up:
                self.db.drop_table(self.table_name)

        self._construct_result_dfs(input_args, results, latencies)

    def lancedb_completion_fn(self, table, embedding_fn, **kwargs):
        return query_builder(table, embedding_fn, **kwargs)

    def _construct_result_dfs(
        self,
        input_args: list[dict[str, object]],
        results: list[dict[str, object]],
        latencies: list[float],
    ):
        r"""
        Construct a few DataFrames that contain all relevant data (i.e. input arguments, results, evaluation metrics).

        This version only extract the most relevant objects returned by LanceDB.

        Args:
             input_args (list[dict[str, object]]): list of dictionaries, where each of them is a set of
                input argument that was passed into the model
             results (list[dict[str, object]]): list of responses from the model
             latencies (list[float]): list of latency measurements
        """
        # `input_arg_df` contains all all input args
        input_arg_df = pd.DataFrame(input_args)
        # `dynamic_input_arg_df` contains input args that has more than one unique values
        dynamic_input_arg_df = _get_dynamic_columns(input_arg_df)

        # `response_df` contains the extracted response (often being the text response)
        response_dict = dict()
        response_dict["top doc ids"] = [self._extract_top_doc_ids(result) for result in results]
        response_dict["distances"] = [self._extract_lancedb_dists(result) for result in results]
        response_dict["documents"] = [self._extract_lancedb_docs(result) for result in results]
        response_df = pd.DataFrame(response_dict)
        # `result_df` contains everything returned by the completion function
        result_df = response_df  # pd.concat([self.response_df, pd.DataFrame(results)], axis=1)

        # `score_df` contains computed metrics (e.g. latency, evaluation metrics)
        self.score_df = pd.DataFrame({"latency": latencies})

        # `partial_df` contains some input arguments, extracted responses, and score
        self.partial_df = pd.concat([dynamic_input_arg_df, response_df, self.score_df], axis=1)
        # `full_df` contains all input arguments, responses, and score
        self.full_df = pd.concat([input_arg_df, result_df, self.score_df], axis=1)

    @staticmethod
    def _extract_top_doc_ids(output: pd.DataFrame) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from LanceDB."""
        return output.to_dict(orient="list")["ids"]

    @staticmethod
    def _extract_lancedb_dists(output: pd.DataFrame) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from LanceDB."""
        return output.to_dict(orient="list")["_distance"]

    @staticmethod
    def _extract_lancedb_docs(output: pd.DataFrame) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from LanceDB."""
        return output.to_dict(orient="list")["text"]
