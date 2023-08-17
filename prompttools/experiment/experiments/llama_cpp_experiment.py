# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Union
import itertools
import logging

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from time import perf_counter

from .experiment import Experiment
from .error import PromptExperimentException
from prompttools.selector.prompt_selector import PromptSelector


class LlamaCppExperiment(Experiment):
    r"""
    Used to experiment across parameters for a local model, supported by LlamaCpp and GGML.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments. For example, ``model_params`` should have string keys,
          with ``list``s being the values.

    Args:
        model_path (List[str]): list of paths to the models that you would like to run
        prompt (List[str] | List[PromptSelector]): list of prompts to test
        model_params (Dict[str, list[object]]): Parameters for initializing the model. The values should be ``list``s.
        call_params: (Dict[str, list[object]]): Parameters for calling the model completion function.
            The values should be ``list``s.
    """

    MODEL_PARAMETERS = (
        "model_path",
        "lora_path",
        "lora_base",
        "n_ctx",
        "n_parts",
        "seed",
        "f16_kv",
        "logits_all",
        "vocab_only",
        "use_mlock",
        "n_threads",
        "n_batch",
        "use_mmap",
        "last_n_tokens_size",
        "verbose",
    )

    CALL_PARAMETERS = (
        "prompt",
        "suffix",
        "max_tokens",
        "temperature",
        "top_p",
        "logprobs",
        "echo",
        "stop",
        "repeat_penalty",
        "top_k",
    )

    DEFAULT = {
        "lora_path": [None],
        "lora_base": [None],
        "n_ctx": [512],
        "n_parts": [-1],
        "seed": [1337],
        "f16_kv": [True],
        "logits_all": [False],
        "vocab_only": [False],
        "use_mlock": [False],
        "n_threads": [None],
        "n_batch": [512],
        "use_mmap": [True],
        "last_n_tokens_size": [64],
        "verbose": [True],
        "suffix": [None],
        "max_tokens": [128],
        "temperature": [0.8],
        "top_p": [0.95],
        "logprobs": [None],
        "echo": [False],
        "stop": [None],
        "repeat_penalty": [1.1],
        "top_k": [40],
    }

    def __init__(
        self,
        model_path: List[str],
        prompt: Union[List[str], List[PromptSelector]],
        model_params: Dict[str, list[object]] = {},
        call_params: Dict[str, list[object]] = {},
    ):
        if Llama is None:
            raise ModuleNotFoundError(
                "Package `llama-cpp-python` is required to be installed to use this experiment."
                "Please use `pip install llama-cpp-python` to install the package"
            )
        self.completion_fn = self.llama_completion_fn
        self.model_params = model_params
        self.call_params = call_params
        self.model_params["model_path"] = model_path

        # If we are using a prompt selector, we need to
        # render the prompts from the selector
        if isinstance(prompt[0], PromptSelector):
            self.call_params["prompt"] = [selector.for_llama() for selector in prompt]
        else:
            self.call_params["prompt"] = prompt

        # Set defaults
        for param in self.MODEL_PARAMETERS:
            if param not in self.model_params:
                self.model_params[param] = self.DEFAULT[param]
        for param in self.CALL_PARAMETERS:
            if param not in self.call_params:
                self.call_params[param] = self.DEFAULT[param]
        self.all_args = self.model_params | self.call_params
        super().__init__()

    @classmethod
    def initialize(cls, test_parameters: dict[str, list], frozen_parameters: dict):
        raise NotImplementedError(
            "`initialize` is currently not compatible with `LlamaCppExperiment`. Please" "use `__init__` directly."
        )

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

    def llama_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        client = params["client"]
        call_params = {k: v for k, v in params.items() if k != "client"}
        response = client(**call_params)
        return response

    def run(
        self,
        runs: int = 1,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For each combination, it will execute `runs` times, default to 1.
        For local models we need to run this in a single thread.
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        results = []
        latencies = []
        for model_combo in self.model_argument_combos:
            client = Llama(**model_combo)
            for call_combo in self.call_argument_combos:
                for _ in range(runs):
                    call_combo["client"] = client
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
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["text"] for choice in output["choices"]][0]

    def _get_model_names(self):
        return [os.path.basename(combo["model_path"]) for combo in self.argument_combos]

    def _get_prompts(self):
        return [combo["prompt"] for combo in self.argument_combos]
