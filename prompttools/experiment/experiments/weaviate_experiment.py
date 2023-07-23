# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the


# LICENSE file in the root directory of this source tree.

import pandas as pd
from typing import Any, Callable, Dict, Optional
import weaviate
import logging

from .experiment import Experiment

VALID_TASKS = [""]


def default_query_builder(
    client: weaviate.Client,
    class_name: str,
    property_names: list[str],
    text_query: str,
):
    near_text_search_operator = {"concepts": [text_query]}
    return client.query.get(class_name, property_names).with_near_text(near_text_search_operator).with_limit(limit=3)


class WeaviateExperiment(Experiment):
    r"""
    Perform an experiment with ``ChromaDB`` to test different embedding functions or retrieval arguments.
    You can query from an existing collection, or create a new one (and insert documents into it) during
    the experiment. If you choose to create a new collection, it will be automatically cleaned up
    as the experiment ends.

    Args:

    """

    def __init__(
        self,
        client: weaviate.Client,
        class_name: str,
        use_existing_data: bool,
        property_names: list[str],
        text_queries: list[str],
        query_builders: dict[str, Callable] = {"default": default_query_builder},
        vectorizers_and_moduleConfigs: Optional[list[tuple[str, dict]]] = None,
        property_definitions: Optional[list[dict]] = None,
        data_objs: Optional[list] = None,
        distance_metrics: Optional[list[str]] = None,
        vectorIndexConfigs: Optional[list[dict]] = None,
    ):
        self.client: weaviate.Client = client
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
        if use_existing_data and data_objs:
            raise RuntimeError("Either use existing data or do not specify `data_objs` for insertion.")
        if not use_existing_data and not data_objs:
            raise RuntimeError("Either use existing data or specify `data_objs` for insertion.")
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
        self.data_objs = data_objs
        self.text_queries = text_queries
        self.query_builders = query_builders
        super().__init__()

    @staticmethod
    def _generate_vectorIndexConfigs(distance_metric: str):
        return {
            "vectorIndexConfig": {
                "distance": distance_metric,
            }
        }

    def weaviate_completion_fn(
        self,
        # weaviate_class: weaviate.clas  .api.Collection,
        **query_params: Dict[str, Any],
    ):
        r"""
        ChromaDB helper function to make request
        """
        results = self.client.query(**query_params)
        return results

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
        self.results = []
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
                    for data_obj in self.data_objs:
                        batch.add_data_object(
                            data_obj,
                            # data_obj: {'property_name1': 'property_value1', 'property_name2': 'property_value2'}
                            self.class_name,
                        )

            # Query
            query_builder = self.query_builders[arg_dict["query_builder_name"]]
            query_obj = query_builder(self.client, self.class_name, self.property_names, arg_dict["text_query"])
            result = query_obj.do()
            self.results.append(result)

            # Clean up
            logging.info("Cleaning up items in Weaviate...")
            if not self.use_existing_data:
                self.client.schema.delete_class(self.class_name)

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

        for combo in self.argument_combos:
            for k, v in combo.items():
                if k in ("vectorizer", "moduleConfig") and (
                    self.vectorizers_and_moduleConfigs is None or len(self.vectorizers_and_moduleConfigs) <= 1
                ):
                    continue
                if k == "vectorIndexConfig" and not self.is_custom_vectorIndexConfigs:
                    continue
                if k == "query_builder_name" and len(self.query_builders) <= 1:
                    continue
                if k not in data:
                    data[k] = []
                data[k].append(v)

        data["top objs"] = [self._extract_responses(result) for result in self.results]

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

    def _extract_responses(self, response: dict) -> list[dict]:
        return response["data"]["Get"][self.class_name]
