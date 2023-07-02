# TODO: Create one for regular OpenAI completions
from typing import Dict, List, Optional
import openai
import logging

from prompttools.experiment.experiment import Experiment


class OpenAICompletionExperiment(Experiment):
    """
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.
    """

    def __init__(
        self,
        model: List[str],
        prompt: List[str],
        suffix: Optional[List[str]] = [None],
        max_tokens: Optional[List[int]] = [float("inf")],
        temperature: Optional[List[float]] = [1.0],
        top_p: Optional[List[float]] = [1.0],
        n: Optional[List[int]] = [1],
        stream: Optional[List[bool]] = [False],
        logprobs: Optional[List[int]] = [None],
        echo: Optional[List[bool]] = [False],
        stop: Optional[List[List[str]]] = [None],
        presence_penalty: Optional[List[float]] = [0],
        frequency_penalty: Optional[List[float]] = [0],
        best_of: Optional[List[int]] = [1],
        logit_bias: Optional[Dict] = [None],
    ):
        self.completion_fn = openai.Completion.create
        self.all_args = []
        self.all_args.append(model)
        self.all_args.append(prompt)
        self.all_args.append(suffix)
        self.all_args.append(max_tokens)
        self.all_args.append(temperature)
        self.all_args.append(top_p)
        self.all_args.append(n)
        self.all_args.append(stream)
        self.all_args.append(logprobs)
        self.all_args.append(echo)
        self.all_args.append(stop)
        self.all_args.append(presence_penalty)
        self.all_args.append(frequency_penalty)
        self.all_args.append(best_of)
        self.all_args.append(logit_bias)
        super().__init__()

    @staticmethod
    def _extract_responses(output) -> str:
        return [choice.text for choice in output.choices]

    @staticmethod
    def _create_args_dict(args) -> Dict[str, object]:
        args = {
            "model": args[0],
            "prompt": args[1],
            "suffix": args[2],
            "max_tokens": args[2],
            "temperature": args[3],
            "top_p": args[4],
            "n": args[5],
            "stream": args[6],
            "logprobs": args[7],
            "echo": args[8],
            "stop": args[9],
            "presence_penalty": args[10],
            "frequency_penalty": args[11],
            "best_of": args[12],
            "logit_bias": args[13],
        }
        return {name: arg for name, arg in args.items() if arg and arg != float("inf")}
