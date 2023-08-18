import itertools
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import Callable, List, Iterable, Optional, Dict, Any

from prompttools.experiment.experiments.error import PromptExperimentException
from prompttools.mock.mock import mock_qdrant_fn

try:
    import qdrant_client
except ImportError:
    qdrant_client = None

from prompttools.experiment import Experiment

VALID_TASKS = [""]


Embedding = List[float]
EmbeddingFn = Callable[[str], Embedding]


class QdrantExperiment(Experiment):
    DEFAULT_DISTANCE = "Cosine"

    def __init__(
        self,
        client: "qdrant_client.QdrantClient",
        collection_name: str,
        embedding_fn: EmbeddingFn,
        vector_size: int,
        documents: Iterable[str],
        queries: Iterable[str],
        collection_params: Optional[Dict[str, List[Any]]] = None,
        query_params: Optional[Dict[str, List[Any]]] = None,
    ):
        if qdrant_client is None:
            raise ModuleNotFoundError(
                "Package `qdrant-client` is required to be installed to use this "
                "experiment. Please use `pip install qdrant-client` to install the "
                "package"
            )
        self.client = client
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.documents = documents
        self.queries = queries
        self.collection_params = collection_params or {}

        if "vectors_config__size" in collection_params:
            warnings.warn(
                "The parameter 'vectors_config__size' is not allowed in "
                "QdrantExperiment. The vector size is determined by the embedding "
                "function. It will be overwritten by {}".format(vector_size)
            )
        # The vector size is a required parameter for Qdrant and has to be overwritten
        self.collection_params["vectors_config__size"] = [vector_size]

        # The distance is set only if not provided by the user
        if "vectors_config__distance" not in collection_params:
            self.collection_params["vectors_config__distance"] = [self.DEFAULT_DISTANCE]

        self.query_params = query_params or {}
        self.completion_fn = self.qdrant_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_qdrant_fn
        self.collection_args_combo: List[dict] = []
        self.query_argument_combos: List[dict] = []
        self.vectorized_documents: List["qdrant_client.models.Record"] = []
        super().__init__()

    @classmethod
    def initialize(cls, test_parameters: dict[str, dict[str, list]], frozen_parameters: dict):
        required_frozen_params = (
            "client",
            "collection_name",
            "embedding_fn",
            "vector_size",
            "documents",
        )
        for arg_name in required_frozen_params:
            if arg_name not in frozen_parameters or arg_name in test_parameters:
                raise RuntimeError(f"'{arg_name}' must be a frozen parameter in QdrantExperiment.")
        return cls(**test_parameters, **frozen_parameters)

    def qdrant_completion_fn(self, **kwargs) -> List["qdrant_client.qdrant_client.types.ScoredPoint"]:
        query_result = self.client.search(self.collection_name, with_vectors=True, with_payload=True, **kwargs)
        return query_result

    def prepare(self) -> None:
        from qdrant_client import models

        # Vectorize the documents and queries
        self.vectorized_documents = [
            models.Record(
                id=i,
                vector=self.embedding_fn(document),
                payload={"document": document},
            )
            for i, document in enumerate(self.documents)
        ]
        self.query_params["query_vector"] = [self.embedding_fn(query) for query in self.queries]

        self.collection_args_combo: list[dict] = []
        for combo in itertools.product(*self.collection_params.values()):
            self.collection_args_combo.append(dict(zip(self.collection_params.keys(), combo)))

        self.query_argument_combos: list[dict] = []
        for combo in itertools.product(*self.query_params.values()):
            self.query_argument_combos.append(dict(zip(self.query_params.keys(), combo)))

    def run(self, runs: int = 1) -> None:
        from qdrant_client import models

        input_args, results, latencies = [], [], []
        if not self.query_argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        for collection_args in self.collection_args_combo:
            collection_args = self._create_nested_object(collection_args)

            try:
                # Create the collection
                self.client.create_collection(self.collection_name, **collection_args)

                # Upload and index the documents
                self.client.upload_records(self.collection_name, self.vectorized_documents)

                # Wait for the collection to be indexed
                while True:
                    collection_info = self.client.get_collection(self.collection_name)
                    if collection_info.status == models.CollectionStatus.GREEN:
                        logging.info("Collection is indexed")
                        break
                    logging.info("Waiting for collection to be indexed...")
                    time.sleep(1)

                # Run the queries
                for query_args in self.query_argument_combos:
                    query_args = self._create_nested_object(query_args)
                    for _ in range(runs):
                        input_args.append(query_args)
                        self.queue.enqueue(self.completion_fn, query_args)

                results.extend(self.queue.get_results())
                latencies.extend(self.queue.get_latencies())
            finally:
                self.client.delete_collection(self.collection_name)

        self._construct_result_dfs(input_args, results, latencies)

    @staticmethod
    def _extract_responses(output: List["qdrant_client.qdrant_client.types.ScoredPoint"]) -> list[str]:
        return [response.payload["document"] for response in output]

    def _create_nested_object(self, args: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Create a nested object using a flat argument dictionary. If a dict key contains
        a double underscore, it is considered a nested object. For example, if the
        argument dictionary contains the key `a__b__c`, this function will return
        a nested object of the form `{"a": {"b": {"c": <value>}}}`.

        If there are multiple keys with the same prefix, the returned object will
        contain a list of nested objects. For example, if the argument dictionary
        contains the keys `a__b__c` and `a__b__d`, this function will return
        a nested object of the form `{"a": {"b": [{"c": <value>}, {"d": <value>}]}}`.

        Args:
            args: A flat argument dictionary

        Returns:
            dict: A nested object
        """
        tree = lambda: defaultdict(tree)

        nested_object = tree()
        for key, value in args.items():
            key_parts = key.split("__")
            current = nested_object
            for key_part in key_parts[:-1]:
                current = current[key_part]
            current[key_parts[-1]] = value

        # Convert defaultdict to dict
        nested_object = json.loads(json.dumps(nested_object))
        return nested_object
