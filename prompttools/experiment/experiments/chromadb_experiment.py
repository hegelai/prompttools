# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
from typing import Any, Callable, Dict, Optional
import chromadb
import itertools
import logging
from prompttools.mock.mock import mock_chromadb_fn
from .experiment import Experiment

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
        chroma_client: chromadb.Client,
        collection_name: str,
        use_existing_collection: bool,
        query_collection_params: dict,
        embedding_fns: list[Callable] = [chromadb.utils.embedding_functions.DefaultEmbeddingFunction],
        embedding_fn_names: list[str] = ["default"],
        add_to_collection_params: Optional[dict] = None,
    ):
        self.chroma_client: chromadb.Client = chroma_client
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
        self.query_args_combo: list[dict] = []
        super().__init__()

    def chromadb_completion_fn(
        self,
        collection: chromadb.api.Collection,
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
        self.query_args_combo: list[dict] = []
        for combo in itertools.product(*self.query_collection_params.values()):
            self.query_args_combo.append(dict(zip(self.query_collection_params.keys(), combo)))

    def run(self, runs: int = 1):
        self.results = []
        if not self.query_args_combo:
            logging.info("Preparing first...")
            self.prepare()
        for emb_fn in self.embedding_fns:
            if self.use_existing_collection:
                collection = self.chroma_client.get_collection(self.collection_name, embedding_function=emb_fn)
            else:  # Creating a new collection and add documents
                collection = self.chroma_client.create_collection(self.collection_name, embedding_function=emb_fn)
                collection.add(**self.add_to_collection_params)
            for query_arg_dict in self.query_args_combo:
                if "query_texts" in query_arg_dict and "query_embeddings" in query_arg_dict:
                    # `query` does not accept both arguments at the same time
                    continue
                for _ in range(runs):
                    self.results.append(self.chromadb_completion_fn(collection, **query_arg_dict))
            # Clean up
            self.chroma_client.delete_collection(self.collection_name)

    def get_table(self, pivot_data: Dict[str, object], pivot_columns: list[str], pivot: bool) -> pd.DataFrame:
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = {}
        # Add scores for each eval fn, including feedback
        for metric_name, evals in self.scores.items():
            if metric_name != "comparison":
                data[metric_name] = evals

        data["embed_fn"] = []
        self.argument_combos: list[dict] = []
        # Add other args as cols if there was more than 1 input
        for i, emb_fn in enumerate(self.embedding_fns):
            for combo in self.query_args_combo:
                arg_combo = {}
                data["embed_fn"].append(self.embedding_fn_names[i])
                arg_combo["embed_fn"] = self.embedding_fn_names[i]
                for k, v in combo.items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)
                    arg_combo[k] = v
                self.argument_combos.append(arg_combo)

        data["top doc ids"] = [self._extract_top_doc_ids(result) for result in self.results]
        data["distances"] = [self._extract_chromadb_dists(result) for result in self.results]
        data["documents"] = [self._extract_chromadb_docs(result) for result in self.results]
        if pivot_data:
            data[pivot_columns[0]] = [str(pivot_data[str(combo[1])][0]) for combo in self.argument_combos]
            data[pivot_columns[1]] = [str(pivot_data[str(combo[1])][1]) for combo in self.argument_combos]
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
    def _extract_top_doc_ids(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from ChromaDB."""
        return output["ids"]

    @staticmethod
    def _extract_chromadb_dists(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from ChromaDB."""
        return output["distances"]

    @staticmethod
    def _extract_chromadb_docs(output: Dict[str, object]) -> list[tuple[str, float]]:
        r"""Helper function to get distances between documents from ChromaDB."""
        return output["documents"]
