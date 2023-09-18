# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
from typing import Any, Callable, Dict, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

import itertools
import logging
from prompttools.mock.mock import mock_chromadb_fn
from time import perf_counter
from .experiment import Experiment
from ._utils import _get_dynamic_columns

VALID_TASKS = [""]


class ChromaDBExperiment(Experiment):
    r"""
    Perform an experiment with ``ChromaDB`` to test different embedding functions or retrieval arguments.
    You can query from an existing collection, or create a new one (and insert documents into it) during
    the experiment. If you choose to create a new collection, it will be automatically cleaned up
    as the experiment ends.

    Args:
        chroma_client (chromadb.Client): ChromaDB client to interact with your database
        collection_name (str): the collection that you will get or create
        use_existing_collection (bool): determines whether to create a new collection or use
            an existing one
        query_collection_params (dict[str, list]): parameters used to query the collection
            Each value is expected to be a list to create all possible combinations
        embedding_fns (list[Callable]): embedding functions to test in the experiment
            by default only uses the default one in ChromaDB
        embedding_fn_names (list[str]): names of the embedding functions
        add_to_collection_params (Optional[dict]): documents or embeddings that will be added to
            the newly created collection
    """

    PARAMETER_NAMES = ["chroma_client"]

    def __init__(
        self,
        chroma_client: "chromadb.Client",
        collection_name: str,
        use_existing_collection: bool,
        query_collection_params: dict,
        embedding_fns: list[Callable] = [
            chromadb.utils.embedding_functions.DefaultEmbeddingFunction if chromadb else None
        ],
        embedding_fn_names: list[str] = ["default"],
        add_to_collection_params: Optional[dict] = None,
    ):
        if chromadb is None:
            raise ModuleNotFoundError(
                "Package `chromadb` is required to be installed to use this experiment."
                "Please use `pip install chromadb` to install the package"
            )
        self.chroma_client: "chromadb.Client" = chroma_client
        self.completion_fn = self.chromadb_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_chromadb_fn
        self.collection_name = collection_name
        self.use_existing_collection = use_existing_collection
        self.query_collection_params = query_collection_params
        self.embedding_fns = embedding_fns
        self.embedding_fn_names = embedding_fn_names
        if len(embedding_fns) != len(embedding_fn_names):
            raise RuntimeError("Please ensure `embedding_fns` and `embedding_fn_names` are aligned.")
        if use_existing_collection and add_to_collection_params:
            raise RuntimeError("You can either use an existing collection or create a new one during the experiment.")
        if not use_existing_collection and add_to_collection_params is None:
            raise RuntimeError("If you choose to create a new collection, you must also add to it.")
        self.add_to_collection_params = add_to_collection_params if add_to_collection_params else {}
        super().__init__()

    @classmethod
    def initialize(cls, test_parameters: dict[str, list], frozen_parameters: dict):
        required_frozen_params = (
            "chroma_client",
            "collection_name",
            "use_existing_collection",
            "query_collection_params",
        )
        for arg_name in required_frozen_params:
            if arg_name not in frozen_parameters or arg_name in test_parameters:
                raise RuntimeError(f"'{arg_name}' must be a frozen parameter in ChromaDBExperiment.")
        frozen_parameters = {k: [v] for k, v in frozen_parameters.items()}
        return cls(**test_parameters, **frozen_parameters)

    def chromadb_completion_fn(
        self,
        collection: "chromadb.api.Collection",
        **query_params: Dict[str, Any],
    ):
        r"""
        ChromaDB helper function to make request
        """
        results = collection.query(**query_params)
        return results

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        for combo in itertools.product(*self.query_collection_params.values()):
            self.argument_combos.append(dict(zip(self.query_collection_params.keys(), combo)))

    def run(self, runs: int = 1):
        input_args = []  # This will be used to construct DataFrame table
        results = []
        latencies = []
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        for i, emb_fn in enumerate(self.embedding_fns):
            if self.use_existing_collection:
                collection = self.chroma_client.get_collection(self.collection_name, embedding_function=emb_fn)
            else:  # Creating a new collection and add documents
                collection = self.chroma_client.create_collection(self.collection_name, embedding_function=emb_fn)
                collection.add(**self.add_to_collection_params)

            for query_arg_dict in self.argument_combos:
                arg_combo = query_arg_dict.copy()
                arg_combo["embed_fn"] = self.embedding_fn_names[i]  # Save embedding function name to combo
                if "query_texts" in query_arg_dict and "query_embeddings" in query_arg_dict:
                    # `query` does not accept both arguments at the same time
                    continue
                for _ in range(runs):
                    input_args.append(arg_combo)
                    start = perf_counter()

                    results.append(self.chromadb_completion_fn(collection, **query_arg_dict))
                    latencies.append(perf_counter() - start)
            # Clean up
            self.chroma_client.delete_collection(self.collection_name)
        self._construct_result_dfs(input_args, results, latencies)

    def _construct_result_dfs(
        self,
        input_args: list[dict[str, object]],
        results: list[dict[str, object]],
        latencies: list[float],
    ):
        r"""
        Construct a few DataFrames that contain all relevant data (i.e. input arguments, results, evaluation metrics).

        This version only extract the most relevant objects returned by ChromaDB.

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
        response_dict["distances"] = [self._extract_chromadb_dists(result) for result in results]
        response_dict["documents"] = [self._extract_chromadb_docs(result) for result in results]
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
    def _extract_top_doc_ids(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get the top document IDs from ChromaDB."""
        return output["ids"][0]

    @staticmethod
    def _extract_chromadb_dists(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get distances between the prompt and documents from ChromaDB."""
        return output["distances"][0]

    @staticmethod
    def _extract_chromadb_docs(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get the top documents from ChromaDB."""
        return output["documents"][0]
