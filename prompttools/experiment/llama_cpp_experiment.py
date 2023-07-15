# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import logging

from llama_cpp import Llama
from time import perf_counter

from .experiment import Experiment
from .error import PromptExperimentException


class LlamaCppExperiment(Experiment):
    r"""
    Experiment for local models.
    """

    PARAMETER_NAMES = (
        "model_path",
        "prompt",
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
        "prompt",  # TODO: Prompt is being added twice, need to remove this instance without breaking
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

    MODEL_PARAMETERS = PARAMETER_NAMES[2:15]
    CALL_PARAMETERS = PARAMETER_NAMES[16:]

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
        self, model_path: List[str], prompt: List[str], **kwargs: Dict[str, object]
    ):
        self.completion_fn = self.llama_completion_fn
        self.all_args = []
        self.all_args.append(model_path)
        self.all_args.append(prompt)
        for param in self.PARAMETER_NAMES[2:]:
            if param in kwargs:
                self.all_args.append(kwargs[param])
            elif param in self.DEFAULT:
                self.all_args.append(self.DEFAULT[param])
            elif param == "prompt":
                self.all_args.append(prompt)
        super().__init__()

    def llama_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        model_params = {k: v for k, v in params.items() if k in self.MODEL_PARAMETERS}
        call_params = {k: v for k, v in params.items() if k in self.CALL_PARAMETERS}
        client = Llama(model_path=params["model_path"], **model_params)
        response = client(**call_params)
        logging.info(response)
        return response

    def run(
        self,
        tagname: Optional[str] = "",
        input_pairs: Optional[Dict[str, Tuple[str, Dict[str, str]]]] = None,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For local models we need to run this single-thread.
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        for combo in self.argument_combos:
            start = perf_counter()
            res = self.completion_fn(
                **self._create_args_dict(combo, tagname, input_pairs)
            )
            self.results.append(res)
            self.scores["latency"] = perf_counter() - start
        if len(self.results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return output
