import os
from typing import Any, Dict, List
import itertools
import logging


from time import perf_counter

from .experiment import Experiment
from .error import PromptExperimentException
from prompttools.selector.prompt_selector import PromptSelector

from .ollama.ollama_api import OllamaAPI

 

class OllamaExperiment(Experiment):
    r"""

    """

    MODEL_PARAMETERS = ( )

    CALL_PARAMETERS = (
          )

    DEFAULT = {
          }

    def __init__(
        self,
        model_url: str,
        prompt: List[str] | List[PromptSelector],
        model_params: Dict[str, list[object]] = {},
        call_params: Dict[str, list[object]] = {},
        model_name = 'llama2',
    ):
        self.completion_fn = self.ollama_completion_fn
        self.ollama_api = OllamaAPI(model_url)
        self.model_params = model_params
        self.call_params = call_params
        # If we are using a prompt selector, we need to
        # render the prompts from the selector
        if isinstance(prompt[0], PromptSelector):
            self.call_params["prompt"] = [selector.for_llama() for selector in prompt]
        else:
            self.call_params["prompt"] = prompt

        self.call_params["model_name"] = model_name

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

    def ollama_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        client = params["client"]
        call_params = {k: v for k, v in params.items() if k != "client"}
        print(call_params)
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
            self.results = []
            for model_combo in self.model_argument_combos:
                client = self.ollama_api.generate
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
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["text"] for choice in output["choices"]][0]

    def _get_model_names(self):
        return [os.path.basename(combo["model_path"]) for combo in self.argument_combos]

    def _get_prompts(self):
        return [combo["prompt"] for combo in self.argument_combos]
