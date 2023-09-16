# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import pandas as pd
from typing import Any, Dict, Optional, Iterable

try:
    import pinecone
except ImportError:
    pinecone = None

import itertools
import logging

# from prompttools.mock.mock import mock_chromadb_fn
from time import perf_counter
from .experiment import Experiment
from ._utils import _get_dynamic_columns

VALID_TASKS = [""]


class PineconeExperiment(Experiment):
    r"""
    Perform an experiment with ``Pinecone`` to test different embedding functions or retrieval arguments.
    You can query from an existing collection, or create a new one (and insert documents into it) during
    the experiment. If you choose to create a new collection, it will be automatically cleaned up
    as the experiment ends.

    Args:
        index_name (str): the index that you will use or create
        use_existing_index (bool): determines whether to create a new collection or use
            an existing one
        query_index_params (dict[str, list]): parameters used to query the collection
            Each value is expected to be a list to create all possible combinations
        create_index_params (Optional[dict]): configuration of the new index (e.g. number of dimensions,
            distance function)
        data (Optional[list]): documents or embeddings that will be added to
            the newly created collection
    """

    def __init__(
        self,
        index_name: str,
        use_existing_index: bool,
        query_index_params: dict,
        create_index_params: Optional[dict] = None,
        data: Optional[list] = None,
    ):
        if pinecone is None:
            raise ModuleNotFoundError(
                "Package `pinecone` is required to be installed to use this experiment."
                "Please use `pip install pinecone-client` to install the package"
            )
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
        self.index_name = index_name
        self.completion_fn = self.pinecone_completion_fn
        # if os.getenv("DEBUG", default=False):
        #     self.completion_fn = mock_chromadb_fn
        self.use_existing_index = use_existing_index
        self.create_index_params: dict = create_index_params if create_index_params else {}
        self.data: list = data if data is not None else []
        self.query_index_params = query_index_params
        if use_existing_index and create_index_params:
            raise RuntimeError("You can either use an existing collection or create a new one during the experiment.")
        if not use_existing_index and data is None:
            raise RuntimeError("If you choose to create a new collection, you must also add to it.")
        super().__init__()

    # @classmethod
    # def initialize(cls, test_parameters: dict[str, list], frozen_parameters: dict):
    #     required_frozen_params = (
    #         "chroma_client",
    #         "collection_name",
    #         "use_existing_collection",
    #         "query_collection_params",
    #     )
    #     for arg_name in required_frozen_params:
    #         if arg_name not in frozen_parameters or arg_name in test_parameters:
    #             raise RuntimeError(f"'{arg_name}' must be a frozen parameter in ChromaDBExperiment.")
    #     frozen_parameters = {k: [v] for k, v in frozen_parameters.items()}
    #     return cls(**test_parameters, **frozen_parameters)

    def pinecone_completion_fn(
        self,
        index: "pinecone.Index",
        **query_params: Dict[str, Any],
    ):
        r"""
        Pinecone helper function to make request
        """
        result = index.query(**query_params)
        return result

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        # TODO: Add support for the case where `create_index_params` are a list, add logic to expand
        for combo in itertools.product(*self.query_index_params.values()):
            self.argument_combos.append(dict(zip(self.query_index_params.keys(), combo)))

    @staticmethod
    def _batch_upsert(pinecone_index: "pinecone.Index", data: Iterable) -> None:
        batch = []
        for d in data:
            batch.append(d)
            if len(batch) == 100:
                pinecone_index.upsert(batch)
                batch = []
        pinecone_index.upsert(batch)

    @staticmethod
    def _wait_for_eventual_consistency(pinecone_index: "pinecone.Index", n_sample: int) -> None:
        i = 0
        while pinecone_index.describe_index_stats()["total_vector_count"] < n_sample:
            i += 1
            print("Waiting for Pinecone's eventual consistency after inserting data.")
            time.sleep(3)
            if i == 20:
                raise TimeoutError("Pinecone has not insert data due to eventual consistency after 1 minute.")

    def run(self, runs: int = 1):
        input_args = []  # This will be used to construct DataFrame table
        results = []
        latencies = []
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()

        # Insert data
        if not self.use_existing_index:
            # TODO: Add support for the case where params are a list , add logic to `prepare`
            pinecone.create_index(self.index_name, **self.create_index_params)
            index = pinecone.Index(self.index_name)
            self._batch_upsert(index, self.data)
        else:
            index = pinecone.Index(self.index_name)

        if not self.use_existing_index:
            self._wait_for_eventual_consistency(index, len(self.data))

        # Query
        for query_arg_dict in self.argument_combos:
            arg_combo = query_arg_dict.copy()
            for _ in range(runs):
                input_args.append(arg_combo)
                start = perf_counter()
                results.append(self.pinecone_completion_fn(index, **query_arg_dict))
                latencies.append(perf_counter() - start)

        # Clean up
        if not self.use_existing_index:
            pinecone.delete_index(self.index_name)
        self._construct_result_dfs(input_args, results, latencies)

    def _construct_result_dfs(
        self,
        input_args: list[dict[str, object]],
        results: list[dict[str, object]],
        latencies: list[float],
    ):
        r"""
        Construct a few DataFrames that contain all relevant data (i.e. input arguments, results, evaluation metrics).

        This version only extract the most relevant objects returned by Pinecone.

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
        response_dict["scores"] = [self._extract_pinecone_scores(result) for result in results]
        response_dict["documents"] = [self._extract_pinecone_docs(result) for result in results]
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
    def _extract_top_doc_ids(output: "pinecone.QueryResponse") -> list[tuple[str, float]]:
        r"""Helper function to get top document IDs from Pinecone."""
        return [match["id"] for match in output["matches"]]

    @staticmethod
    def _extract_pinecone_scores(output: "pinecone.QueryResponse") -> list[tuple[str, float]]:
        r"""Helper function to get the scores of documents from Pinecone."""
        return [match["score"] for match in output["matches"]]

    @staticmethod
    def _extract_pinecone_docs(output: "pinecone.QueryResponse") -> list[tuple[str, float]]:
        r"""Helper function to get top documents from Pinecone."""
        return [match["values"] for match in output["matches"]]
