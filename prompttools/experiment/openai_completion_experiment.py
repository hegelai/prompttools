# TODO: Create one for regular OpenAI completions
from typing import Callable, Dict, List, Optional
import openai
import pandas as pd
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
        self.completion_fn = openai.ChatCompletion.create
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

    @staticmethod
    def _extract_responses(output) -> str:
        return [choice.text for choice in output.choices]

    @staticmethod
    def _create_args_dict(args) -> Dict[str, object]:
        return {
            "suffix": args[0],
            "max_tokens": args[1],
            "temperature": args[2],
            "top_p": args[3],
            "n": args[4],
            "stream": args[5],
            "logprobs": args[6],
            "echo": args[7],
            "stop": args[8],
            "presence_penalty": args[9],
            "frequency_penalty": args[10],
            "best_of": args[11],
            "logit_bias": args[12],
        }

    def run(self):
        """
        Create tuples of input and output for every possible combination of arguments.
        """
        if not self.argument_combos:
            logging.warning("Please run `prepare` first.")
        self.results = []
        for combo in self.argument_combos:
            self.result.append(
                openai.Completion.create(
                    model=combo[0],
                    prompt=combo[1],
                    suffix=combo[2],
                    max_tokens=combo[3],
                    temperature=combo[4],
                    top_p=combo[5],
                    n=combo[6],
                    stream=combo[7],
                    logprobs=combo[8],
                    echo=combo[9],
                    stop=combo[10],
                    presence_penalty=combo[10],
                    frequency_penalty=combo[11],
                    best_of=combo[12],
                    logit_bias=combo[13],
                )
            )

    def evaluate(self, eval_fn: Callable):
        """
        Using the given evaluation function, all input/response pairs are evaluated.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
        self.scores = []
        for i, result in enumerate(self.results):
            # Pass the messages and results into the eval function
            self.scores.append(
                eval_fn(self.argument_combos[i][1], self._extract_responses(result))
            )

    def get_table(self):
        return pd.DataFrame(
            {
                "model": [combo[0] for combo in self.argument_combos],
                "prompt": [combo[1] for combo in self.argument_combos],
                "response(s)": [
                    self._extract_responses(result) for result in self.results
                ],
                "score": self.scores,
                "other": [
                    self._create_args_dict(combo[2:]) for combo in self.argument_combos
                ],
            }
        )
