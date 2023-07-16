# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict
import chromadb

from prompttools.mock.mock import mock_chromadb_fn
from .experiment import Experiment

VALID_TASKS = [""]


class ChromaDBExperiment(Experiment):
    r"""
    Experiment for ChromaDB.
    """

    PARAMETER_NAMES = ["chroma_client"]

    def __init__(
        self,
        chroma_client: chromadb.Client,
        **kwargs: Dict[str, object],
    ):
        self.chroma_client = chroma_client
        self.completion_fn = self.chromadb_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_chromadb_fn
        self.all_args = []
        self.all_args.append(chroma_client)
        for k, v in kwargs.items():
            self.PARAMETER_NAMES.append(k)
            self.all_args.append(v)
        super().__init__()
    
    def chromadb_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        ChromaDB helper function to make request
        """
        client = self.chroma_client
        collection = client.create_collection(name=params["collection"])
        collection.add(
            documents=params["documents"],
            metadatas=params["metadatas"],
            ids=params["ids"]
        )
        results = collection.query(
            query_texts=params["queries"],
            n_results=params["n_results"]
        )
        return results

    @staticmethod
    def _extract_chromadb_dists(output: Dict[str, object]) -> list[str]:
        "Helper function to get distances between documents from ChromaDB."
        return output