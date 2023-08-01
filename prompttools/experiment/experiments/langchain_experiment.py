# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# from pydantic.main import ModelMetaclass

from time import perf_counter
import logging

from prompttools.mock.mock import mock_lc_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException

VALID_TASKS = ()


class SequentialChainExperiment(Experiment):
    """
    Experiment for LangChain sequential chains.
    For sequential chains, the order in which
    links in the chain are submitted to the
    class is very strict. Users should take extra
    precaution since this is not yet enforced.
    """

    MODEL_PARAMETERS = ["temperature"]
    CALL_PARAMETERS = {"prompt_chain", "temperature", "input_variables", "output_variables", "output_key", "prompt"}

    def __init__(
        self,
        llms: List[Any],
        **kwargs: Dict[str, object],
    ):
        self.completion_fn = self.lc_completion_fn
        self.prompt = kwargs["prompt"]
        self.call_params = {k: kwargs[k] for k in self.CALL_PARAMETERS if k in self.CALL_PARAMETERS}
        self.output_keys = kwargs["output_key"]
        self.llms = []
        # Only supporting temperature for now
        for llm in llms:
            for temp in kwargs["temperature"]:
                self.llms.append(llm(temperature=temp))
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_lc_completion_fn
        self.chain_params = list(zip(kwargs["prompt_chain"], kwargs["output_key"], strict=True))
        self.all_args = {
            "temperature": kwargs["temperature"],
            "llms": llms,
            "prompt": self.prompt
        }
        super().__init__()

    def prepare_chain(self) -> None:
        """
        Links each individual chain to assemble
        a sequential chain.
        """
        self.seq_chains = []
        for llm in self.llms:
            seq_chain = [
                LLMChain(llm=llm, prompt=prompt, output_key=output_key)
                for prompt, output_key in self.chain_params
                ]
            overall_chain = SequentialChain(
                chains=seq_chain,
                input_variables=self.call_params["input_variables"],
                output_variables=self.call_params["output_variables"],
                verbose=True,
            )
            self.seq_chains.append(overall_chain)

    def lc_completion_fn(
        self,
        seq_chain,
    ):
        r"""
        Makes request to LLM using the assembled
        sequential chain.
        """
        return seq_chain(self.prompt)

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
            self.prepare_chain()
        self.results = []
        for seq_chain in self.seq_chains:
            for _ in range(runs):
                start = perf_counter()
                res = self.completion_fn(seq_chain)
                self.scores["latency"].append(perf_counter() - start)
                self.results.append(res)
                self.argument_combos.append({
                    "llm": seq_chain.chains[0].llm,
                    "temperature": seq_chain.chains[0].llm.temperature,
                    "prompt": self.prompt,
                    })
        if len(self.results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException

    @staticmethod
    def _extract_responses(output: List[Dict[str, object]]) -> list[str]:
        return [str(output)]


class RouterChainExperiment(Experiment):
    """
    Experiment for LangChain router chains.
    """
    # TODO: functionality for router chains
