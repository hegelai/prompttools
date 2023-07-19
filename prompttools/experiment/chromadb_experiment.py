# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
from typing import Any, Dict, Optional
import chromadb
import itertools
import logging
from prompttools.mock.mock import mock_chromadb_fn
from .experiments.experiment import Experiment

VALID_TASKS = [""]


class ChromaDBExperiment(Experiment):
    r"""
    Experiment for ChromaDB.
    """

    PARAMETER_NAMES = ["chroma_client"]

    def __init__(
        self,
        chroma_client: chromadb.Client,
        collection_name: str,
        query_collection_params: dict,
        get_collection_params: Optional[dict] = None,
        # get_or_create_collection_params: Optional[dict] = None,
        # add_to_collection_params: Optional[dict] = None,  # TODO: Implement add collection parameters
    ):
        self.chroma_client: chromadb.Client = chroma_client
        self.completion_fn = self.chromadb_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_chromadb_fn
        self.collection_name = collection_name
        self.get_collection_params = (
            get_collection_params if get_collection_params else {}
        )

        # self.get_or_create_collection_params = (
        #     get_or_create_collection_params if get_or_create_collection_params else {}
        # )
        # self.add_to_collection_params = (
        #     add_to_collection_params if add_to_collection_params else {}
        # )
        self.query_collection_params = query_collection_params
        self.collection = None
        self.query_args_combo: list[dict] = []
        # TODO: Might need to separate `add` collection and `create` collection cases
        super().__init__()

    # TODO: If a collection is created during the experiment, it needs to be deleted
    def chromadb_completion_fn(
        self,
        **query_params: Dict[str, Any],
    ):
        r"""
        ChromaDB helper function to make request
        """
        collection = self.chroma_client.get_or_create_collection(
            self.collection_name, **self.get_collection_params
        )

        results = collection.query(**query_params)
        return results

    def prepare(self) -> None:
        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name, **self.get_collection_params
        )
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.query_args_combo: list[dict] = []
        for combo in itertools.product(*self.query_collection_params.values()):
            self.query_args_combo.append(
                dict(zip(self.query_collection_params.keys(), combo))
            )

    def run(self, runs: int = 1):
        self.results = []
        if not self.query_args_combo:
            logging.info("Preparing first...")
            self.prepare()
        for query_arg_dict in self.query_args_combo:
            if "query_texts" in query_arg_dict and "query_embeddings" in query_arg_dict:
                # `query` does not accept both arguments at the same time
                continue
            for _ in range(runs):
                self.results.append(self.chromadb_completion_fn(**query_arg_dict))

    def get_table(
        self, pivot_data: Dict[str, object], pivot_columns: list[str], pivot: bool
    ) -> pd.DataFrame:
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = {}
        # Add scores for each eval fn, including feedback
        for metric_name, evals in self.scores.items():
            if metric_name != "comparison":
                data[metric_name] = evals
        # Add other args as cols if there was more than 1 input
        for combo in self.query_args_combo:
            for k, v in combo.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

        data["top doc ids"] = [
            self._extract_top_doc_ids(result) for result in self.results
        ]
        data["distances"] = [
            self._extract_chromadb_dists(result) for result in self.results
        ]

        if pivot_data:
            data[pivot_columns[0]] = [
                str(pivot_data[str(combo[1])][0]) for combo in self.argument_combos
            ]
            data[pivot_columns[1]] = [
                str(pivot_data[str(combo[1])][1]) for combo in self.argument_combos
            ]
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
