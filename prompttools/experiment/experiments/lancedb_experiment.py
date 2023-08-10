# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import itertools
import pandas as pd
from typing import Callable, Dict, Optional
from collections import defaultdict

try:
    import lancedb
    from lancedb.embeddings import with_embeddings

except ImportError:
    lancedb = None

import logging
from .experiment import Experiment

VALID_TASKS = [""]

def query_builder(
    table: "lancedb.Table",
    embed_fn: Callable,
    text: str,
    metric:str = "cosine",
    limit: int = 3,
    filter: str = None,
):  
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
        embedding_fns: Dict,
        query_args: Dict,
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
        self.all_args = query_args
        self.clean_up = clean_up
        super().__init__()
    
    def prepare(self):
        all_args = self.all_args.copy()
        all_args.update({"embed_fn": list(self.embedding_fns.keys())})
        # Take embedding functions into account when creating argument combinations but treat it separately
        # as it is a mandatory argument(?)
        self.argument_combos = [dict(zip(all_args.keys(), val)) for val in itertools.product(*all_args.values())]

    def run(self, runs: int = 1):
        self.results = []
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        for query_arg_dict in self.argument_combos:
            query_args = query_arg_dict.copy()
            emb_fn_name = query_args.pop("embed_fn")
            emb_fn = self.embedding_fns[emb_fn_name]
            
            if self.use_existing_table:
                table = self.db.open_table(self.table_name)
                if not table:
                    raise RuntimeError(f"Table {self.table_name} does not exist.")
            else:
                data = with_embeddings(emb_fn, self.data, "text")
                table = self.db.create_table(self.table_name, data, mode="overwrite")
            for _ in range(runs):
                self.results.append(self.lancedb_completion_fn(table=table, embedding_fn=emb_fn, **query_args))
            # Clean up
            if self.clean_up:
                self.db.drop_table(self.table_name)

    def lancedb_completion_fn(self, table, embedding_fn,**kwargs):
        return query_builder(table, embedding_fn, **kwargs)

    def get_table(self, pivot_data: Dict[str, object], pivot_columns: list[str], pivot: bool) -> pd.DataFrame:
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = defaultdict(list)
        # Add scores for each eval fn, including feedback
        for metric_name, evals in self.scores.items():
            if metric_name != "comparison":
                data[metric_name] = evals

        self.result_argument_combos: list[dict] = []
        for combo in self.argument_combos:
            arg_combo = {}
            for k, v in combo.items():
                data[k].append(v)
                arg_combo[k] = v
            self.result_argument_combos.append(arg_combo)

        data["top doc ids"] = [result["ids"].to_list() for result in self.results]
        data["distances"] = [result["score"].to_list() for result in self.results]
        data["documents"] = [result["text"].to_list() for result in self.results]
        data["embed_fn"] = [combo["embed_fn"] for combo in self.argument_combos]
        if pivot_data:
            data[pivot_columns[0]] = [str(pivot_data[str(combo[1])][0]) for combo in self.result_argument_combos]
            data[pivot_columns[1]] = [str(pivot_data[str(combo[1])][1]) for combo in self.result_argument_combos]
        df = pd.DataFrame(data)
        if pivot:
            df = pd.pivot_table(
                df,
                values="top doc ids",
                index=[pivot_columns[1]],
                columns=[pivot_columns[0]],
                aggfunc=lambda x: x.iloc[0],
            )
        return df

    
    @staticmethod
    def _extract_responses(output: pd.DataFrame) -> dict:
        r"""Helper function to get distances between documents from LanceDB."""
        return output.to_dict(orient="list")
