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
        "prompt",
        "suffix",
        "max_tokens",
        "temperature",
        "top_p",
        "logprobs",
        "echo",
        "stop_sequences",
        "repeat_penalty",
        "top_k",
    )

    MODEL_PARAMETERS = PARAMETER_NAMES[2:15]
    CALL_PARAMETERS = PARAMETER_NAMES[16:]

    def __init__(
        self,
        model_path: List[str],
        prompt: List[str],
        use_scribe: bool = False,
        scribe_name: str = "HuggingFace Experiment",
        **kwargs: Dict[str,object]
    ):
        self.use_scribe = use_scribe
        if use_scribe:
            from hegel.scribe import HegelScribe

            self.completion_fn = HegelScribe(
                name=scribe_name, completion_fn=self.llama_completion_fn
            )
        else:
            self.completion_fn = self.llama_completion_fn
        
        self.all_args = []
        self.all_args.append(model_path)
        self.all_args.append(prompt)
        for param in self.PARAMETER_NAMES[2:]:
            if param in kwargs:
                self.all_args.append(kwargs[param])
        super().__init__()
    
    def llama_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        print("Making call")
        model_params = {k: v for k,v in params.items() if k in self.MODEL_PARAMETERS}
        call_params = {k: v for k,v in params.items() if k in self.CALL_PARAMETERS}
        client = Llama(params['model_path'], **model_params)
        print("Got client")
        response = client(prompt=params["prompt"], **call_params)
        print("Made call")
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
            res = self.completion_fn(**self._create_args_dict(combo, tagname, input_pairs))
            self.results.append(res)
            self.scores["latency"] = perf_counter() - start
        if len(self.results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return output