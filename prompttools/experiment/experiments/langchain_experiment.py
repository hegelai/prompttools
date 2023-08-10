# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List
import itertools

from time import perf_counter
import logging

try:
    from langchain.chains import LLMChain, SequentialChain
except ImportError:
    LLMChain = None

from prompttools.mock.mock import mock_lc_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException


class SequentialChainExperiment(Experiment):
    r"""
    Testing Lang Chain sequential chains.
    """

    MODEL_PARAMETERS = ["llm"]

    CALL_PARAMETERS = ["prompt_template", "prompt"]

    def __init__(
        self,
        llm: List[Any],
        prompt_template: List[List[Any]],
        prompt: List[str],
        **kwargs: Dict[str, object],
    ):
        if LLMChain is None:
            raise ModuleNotFoundError(
                "Package `langchain` is required to be installed to use this experiment."
                "Please use `pip install langchain` to install the package"
            )
        self.completion_fn = self.lc_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_lc_completion_fn
        self.model_params = dict(llm=llm)  # placeholder for future

        self.call_params = dict(prompt_template=prompt_template, prompt=prompt)
        for k, v in kwargs.items():
            self.CALL_PARAMETERS.append(k)
            self.call_params[k] = v

        self.all_args = self.model_params | self.call_params
        super().__init__()

    def prepare(self) -> None:
        r"""
        Combo builder.
        """
        self.model_argument_combos = [
            dict(zip(self.model_params, val, strict=False)) for val in itertools.product(*self.model_params.values())
        ]
        self.call_argument_combos = [
            dict(zip(self.call_params, val, strict=False)) for val in itertools.product(*self.call_params.values())
        ]

    def lc_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request.
        """
        client = params["client"]
        response = client(params["prompt"])
        return response

    def run(
        self,
        runs: int = 1,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For each combination, it will execute `runs` times, default to 1.
        # TODO This can be done with an async queue.
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        self.results = []
        for model_combo in self.model_argument_combos:
            for call_combo in self.call_argument_combos:
                llm = model_combo["llm"]
                llm = llm(temperature=call_combo["temperature"])
                chain = []
                for i, prompt_template in enumerate(call_combo["prompt_template"]):
                    chain.append(LLMChain(llm=llm, prompt=prompt_template, output_key=call_combo["output_key"][i]))
                client = SequentialChain(
                    chains=chain,
                    input_variables=call_combo["input_variables"],
                    output_variables=call_combo["output_variables"],
                    verbose=True,
                )
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
    def _extract_responses(output: List[Dict[str, object]]) -> str:
        return str({k: output[k] for k in output})
