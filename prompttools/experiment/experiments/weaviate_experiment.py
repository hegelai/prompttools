# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the


# LICENSE file in the root directory of this source tree.

import pandas as pd
from typing import Callable, Optional

try:
    import weaviate
except ImportError:
    weaviate = None
import logging

from time import perf_counter
from .experiment import Experiment
from ._utils import _get_dynamic_columns

VALID_TASKS = [""]


def default_query_builder(
    client: "weaviate.Client",
    class_name: str,
    property_names: list[str],
    text_query: str,
):
    near_text_search_operator = {"concepts": [text_query]}
    return client.query.get(class_name, property_names).with_near_text(near_text_search_operator).with_limit(limit=3)


class WeaviateExperiment(Experiment):
    r"""
    Perform an experiment with Weaviate to test different vectorizers or querying functions.
    You can query from an existing class, or create a new one (and insert data objects into it) during
    the experiment. If you choose to create a new class, it will be automatically cleaned up
    as the experiment ends.

    Args:
        client (weaviate.Client): The Weaviate client instance to interact with the Weaviate server.
        class_name (str): The name of the Weaviate class (equivalent to a collection in ChromaDB).
        use_existing_data (bool): If ``True``, indicates that existing data will be used for the experiment.
            If ``False``, new data objects will be inserted into Weaviate during the experiment.
        property_names (list[str]): List of property names in the Weaviate class to be used in the experiment.
        text_queries (list[str]): List of text queries to be used for retrieval in the experiment.
        query_builders (dict[str, Callable], optional): A dictionary containing different query builders.
            The key should be the name of the function for visualization purposes.
            The value should be a Callable function that constructs and returns a Weaviate query object.
            Defaults to a built-in query function.
        vectorizers_and_moduleConfigs (Optional[list[tuple[str, dict]]], optional): List of tuples, where each tuple
            contains the name of the vectorizer and its corresponding moduleConfig as a dictionary. This
            is used during data insertion (if necessary).
        property_definitions (Optional[list[dict]], optional): List of property definitions for the Weaviate class.
            Each property definition is a dictionary containing the property name and data type.
            This is used during data insertion (if necessary).
        data_objects (Optional[list], optional): List of data objects to be inserted into Weaviate during the
            experiment. Each data object is a dictionary representing the property-value pairs.
        distance_metrics (Optional[list[str]], optional): List of distance metrics to be used in the experiment.
            These metrics will be used for generating vectorIndexConfig. This is used to define the class object.
            If necessary, either use ``distance_metrics`` or ``vectorIndexConfigs``, not both.
        vectorIndexConfigs (Optional[list[dict]], optional): List of vectorIndexConfig to be used in the
            experiment to define the class object.

    Note:
        - If ``use_existing_data`` is ``False``, the experiment will create a new Weaviate class and insert
          ``data_objects`` into it. The class and ``data_objects`` will be automatically cleaned up at the end of the
          experiment.
        - Either use existing data or specify ``data_objs`` and ``vectorizers`` for insertion.
        - Either ``distance_metrics`` or ``vectorIndexConfigs`` should be provided if necessary, not both.
        - If you pass in a custom ``query_builder`` function, it should accept the same parameters as
          `the default one as seen here`_.

    .. _the default one as seen here:
        https://github.com/hegelai/prompttools/blob/main/prompttools/experiment/experiments/weaviate_experiment.py
    """

    def __init__(
        self,
        client: "weaviate.Client",
        class_name: str,
        use_existing_data: bool,
        property_names: list[str],
        text_queries: list[str],
        query_builders: dict[str, Callable] = {"default": default_query_builder},
        vectorizers_and_moduleConfigs: Optional[list[tuple[str, dict]]] = None,
        property_definitions: Optional[list[dict]] = None,
        data_objects: Optional[list] = None,
        distance_metrics: Optional[list[str]] = None,
        vectorIndexConfigs: Optional[list[dict]] = None,
    ):
        if weaviate is None:
            raise ModuleNotFoundError(
                "Package `weaviate` is required to be installed to use this experiment."
                "Please use `pip install weaviate-client` to install the package"
            )
        self.client = client
        self.completion_fn = self.weaviate_completion_fn
        # TODO: Add a mock function if needed
        # if os.getenv("DEBUG", default=False):
        #     self.completion_fn = mock_chromadb_fn
        self.class_name = class_name
        self.vectorizers_and_moduleConfigs = vectorizers_and_moduleConfigs
        self.property_names = property_names
        self.property_definitions = property_definitions
        if distance_metrics and vectorIndexConfigs:
            raise RuntimeError("Either use `distance_metrics` or `vectorIndexConfigs`.")
        if use_existing_data and data_objects:
            raise RuntimeError("Either use existing data or do not specify `data_objs` for insertion.")
        if not use_existing_data and not data_objects:
            raise RuntimeError("Either use existing data or specify `data_objs` for insertion.")
        if not use_existing_data and not vectorizers_and_moduleConfigs:
            raise RuntimeError("Either use existing data or specify `vectorizers_and_moduleConfigs` for insertion.")
        self.use_existing_data = use_existing_data
        self.is_custom_vectorIndexConfigs = vectorIndexConfigs or distance_metrics
        if vectorIndexConfigs:
            self.vectorIndexConfigs = vectorIndexConfigs
        elif distance_metrics:
            self.vectorIndexConfigs = [
                self._generate_vectorIndexConfigs(dist_metric) for dist_metric in distance_metrics
            ]
        else:  # weaviate's default
            self.vectorIndexConfigs = [self._generate_vectorIndexConfigs("cosine")]
        self.data_objects = data_objects
        self.text_queries = text_queries
        self.query_builders = query_builders
        super().__init__()

    @classmethod
    def initialize(cls, test_parameters: dict[str, list], frozen_parameters: dict):
        required_frozen_params = ("client", "class_name", "use_existing_data")
        for arg_name in required_frozen_params:
            if arg_name not in frozen_parameters or arg_name in test_parameters:
                raise RuntimeError(f"'{arg_name}' must be a frozen parameter in WeaviateExperiment.")
        frozen_parameters = {k: [v] for k, v in frozen_parameters.items()}
        return cls(**test_parameters, **frozen_parameters)

    @staticmethod
    def _generate_vectorIndexConfigs(distance_metric: str):
        return {
            "vectorIndexConfig": {
                "distance": distance_metric,
            }
        }

    def weaviate_completion_fn(
        self,
        query_builder: Callable,
        text_query: str,
    ):
        r"""
        Weaviate helper function to make request
        """
        query_obj = query_builder(self.client, self.class_name, self.property_names, text_query)
        return query_obj.do()

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos: list[dict] = []
        vectorizers_and_moduleConfigs = (
            [("", {})] if self.vectorizers_and_moduleConfigs is None else self.vectorizers_and_moduleConfigs
        )
        for vectorizer, moduleConfig in vectorizers_and_moduleConfigs:
            for vectorIndexConfig in self.vectorIndexConfigs:
                for query_builder_name, query_builder in self.query_builders.items():
                    for text_query in self.text_queries:
                        self.argument_combos.append(
                            {
                                "vectorizer": vectorizer,
                                "moduleConfig": moduleConfig,
                                "vectorIndexConfig": vectorIndexConfig,
                                "text_query": text_query,
                                "query_builder_name": query_builder_name,
                            }
                        )

    def run(self, runs: int = 1):
        results = []
        latencies = []
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        for arg_dict in self.argument_combos:
            if not self.use_existing_data:
                # Create class (equivalent to collection in ChromaDB)
                weaviate_class_obj = {
                    "class": self.class_name,
                    "vectorizer": arg_dict["vectorizer"],
                    "moduleConfig": arg_dict["moduleConfig"],
                    "properties": self.property_definitions,
                    "vectorIndexConfig": arg_dict["vectorIndexConfig"],
                }
                self.client.schema.create_class(weaviate_class_obj)

                # Batch Insert Items
                logging.info("Inserting items into Weaviate...")
                with self.client.batch() as batch:
                    for data_obj in self.data_objects:
                        batch.add_data_object(
                            data_obj,
                            # data_obj: {'property_name1': 'property_value1', 'property_name2': 'property_value2'}
                            self.class_name,
                        )

            # Query
            query_builder = self.query_builders[arg_dict["query_builder_name"]]
            for _ in range(runs):
                start = perf_counter()
                result = self.completion_fn(query_builder, arg_dict["text_query"])
                latencies.append(perf_counter() - start)
                results.append(result)

            # Clean up
            logging.info("Cleaning up items in Weaviate...")
            if not self.use_existing_data:
                self.client.schema.delete_class(self.class_name)
        self._construct_result_dfs([c for c in self.argument_combos for _ in range(runs)], results, latencies)

    # TODO: Collect and add latency
    def _construct_result_dfs(
        self,
        input_args: list[dict[str, object]],
        results: list[dict[str, object]],
        latencies: list[float],
    ):
        r"""
        Construct a few DataFrames that contain all relevant data (i.e. input arguments, results, evaluation metrics).

        This version only extract the most relevant objects returned by Weaviate.

        Args:
             input_args (list[dict[str, object]]): list of dictionaries, where each of them is a set of
                input argument that was passed into the model
             results (list[dict[str, object]]): list of responses from the model
        """
        # `input_arg_df` contains all all input args
        input_arg_df = pd.DataFrame(input_args)
        # `dynamic_input_arg_df` contains input args that has more than one unique values
        dynamic_input_arg_df = _get_dynamic_columns(input_arg_df)

        # `response_df` contains the extracted response (often being the text response)
        response_dict = dict()
        response_dict["top objs"] = [self._extract_responses(result) for result in results]
        response_df = pd.DataFrame(response_dict)
        # `result_df` contains everything returned by the completion function
        result_df = response_df  # pd.concat([self.response_df, pd.DataFrame(results)], axis=1)

        # `score_df` contains computed metrics (e.g. latency, evaluation metrics)
        self.score_df = pd.DataFrame({"latency": latencies})

        # `partial_df` contains some input arguments, extracted responses, and score
        self.partial_df = pd.concat([dynamic_input_arg_df, response_df, self.score_df], axis=1)
        # `full_df` contains all input arguments, responses, and score
        self.full_df = pd.concat([input_arg_df, result_df, self.score_df], axis=1)

    def _extract_responses(self, response: dict) -> list[dict]:
        return response["data"]["Get"][self.class_name]
