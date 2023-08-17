# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Tuple
import itertools
from time import perf_counter
import logging

try:
    from mysql.connector.connection_cext import CMySQLConnection
except ImportError:
    CMySQLConnection = None

from prompttools.mock.mock import mock_mindsdb_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException


class MindsDBExperiment(Experiment):
    r"""
    An experiment class for MindsDB.
    This accepts combinations of MindsDB inputs to form SQL queries, returning a list of responses.

    Args:
        db_connector (CMySQLConnection): Connector MindsDB
        kwargs (dict): keyword arguments for the model
    """

    def __init__(
        self,
        db_connector: "CMySQLConnection",
        **kwargs: Dict[str, object],
    ):
        self.cursor = db_connector.cursor()
        self.completion_fn = self.mindsdb_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_mindsdb_completion_fn

        self.call_params = dict(prompt=kwargs["prompt"])
        self.model_params = dict({k: kwargs[k] for k in kwargs if k != "prompt"})

        self.all_args = self.model_params | self.call_params
        super().__init__()

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.model_argument_combos = [
            dict(zip(self.model_params, val)) for val in itertools.product(*self.model_params.values())
        ]
        self.call_argument_combos = [
            dict(zip(self.call_params, val)) for val in itertools.product(*self.call_params.values())
        ]

    def mindsdb_completion_fn(
        self,
        **params: Dict[str, Any],
    ) -> List[Any]:
        r"""
        MindsDB helper function to make request.
        """
        prompt = params["prompt"]

        self.cursor.execute(prompt)
        return [x for x in self.cursor]

    def run(
        self,
        runs: int = 1,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For each combination, it will execute `runs` times, default to 1.
        # TODO This can be done with an async queue
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        results = []
        latencies = []
        for model_combo in self.model_argument_combos:
            for call_combo in self.call_argument_combos:
                call_combo["prompt"] = call_combo["prompt"].format(
                    table=model_combo["table"],
                    author_username=model_combo["author_username"],
                    text=model_combo["text"],
                )
                for _ in range(runs):
                    call_combo["client"] = self.cursor
                    start = perf_counter()
                    res = self.completion_fn(**call_combo)
                    latencies.append(perf_counter() - start)
                    results.append(res)
                    self.argument_combos.append(model_combo | call_combo)
        if len(results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException
        self._construct_result_dfs(self.argument_combos, results, latencies, extract_response_equal_full_result=True)

    @staticmethod
    def _extract_responses(output: List[Dict[str, object]]) -> Tuple[str]:
        return output[0]
