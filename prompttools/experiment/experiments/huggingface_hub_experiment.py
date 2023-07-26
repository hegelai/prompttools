# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List
import itertools
from huggingface_hub.inference_api import InferenceApi
from time import perf_counter
import logging

from prompttools.mock.mock import mock_hf_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException

VALID_TASKS = ("text2text-generation", "text-generation", "summarization")


class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    It accepts lists for each argument passed into Hugging Face Hub's API,
    then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
    """

    MODEL_PARAMETERS = ["repo_id", "task"]

    CALL_PARAMETERS = ["prompt"]

    def __init__(
        self,
        repo_id: List[str],
        prompt: List[str],
        task: List[str] = ["text-generation"],
        **kwargs: Dict[str, object],
    ):
        self.completion_fn = self.hf_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_hf_completion_fn
        self.model_params = dict(repo_id=repo_id, task=task)

        self.call_params = dict(prompt=prompt)
        for k, v in kwargs.items():
            self.CALL_PARAMETERS.append(k)
            self.call_params[k] = v

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

    def hf_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        client = params["client"]
        model_kwargs = {k: params[k] for k in params if k not in ["client", "prompt"]}
        response = client(inputs=params["prompt"], params=model_kwargs)
        return response

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
        self.results = []
        for model_combo in self.model_argument_combos:
            client = InferenceApi(
                repo_id=model_combo["repo_id"],
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                task=model_combo["task"],
            )
            for call_combo in self.call_argument_combos:
                for _ in range(runs):
                    call_combo["client"] = client
                    start = perf_counter()
                    res = self.completion_fn(**call_combo)
                    self.scores["latency"].append(perf_counter() - start)
                    self.results.append(res)
                    self.argument_combos.append(model_combo | call_combo)
        if len(self.results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException

    @staticmethod
    def _extract_responses(output: List[Dict[str, object]]) -> list[str]:
        return [choice["generated_text"] for choice in output][0]
