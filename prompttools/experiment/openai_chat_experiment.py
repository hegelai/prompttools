from typing import Callable, Dict, List, Optional
import itertools
import openai
import pandas as pd
import logging
from utils import is_interactive
from IPython import display
from tabulate import tabulate

logging.getLogger().setLevel(logging.INFO)


class OpenAIChatExperiment:
    """
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.
    """

    def __init__(
        self,
        model: List[str],
        messages: List[List[Dict[str, str]]],
        temperature: Optional[List[float]] = [1.0],
        top_p: Optional[List[float]] = [1.0],
        n: Optional[List[int]] = [1],
        stream: Optional[List[bool]] = [False],
        stop: Optional[List[List[str]]] = [None],
        max_tokens: Optional[List[int]] = [float("inf")],
        presence_penalty: Optional[List[float]] = [0],
        frequency_penalty: Optional[List[float]] = [0],
        logit_bias: Optional[Dict] = [None],
    ):
        self.completion_fn = openai.ChatCompletion.create
        self.all_args = []
        self.all_args.append(model)
        self.all_args.append(messages)
        self.all_args.append(temperature)
        self.all_args.append(top_p)
        self.all_args.append(n)
        self.all_args.append(stream)
        self.all_args.append(stop)
        self.all_args.append(max_tokens)
        self.all_args.append(presence_penalty)
        self.all_args.append(frequency_penalty)
        self.all_args.append(logit_bias)

    @staticmethod
    def _extract_chat_responses(output) -> str:
        return [choice.message.content for choice in output.choices]

    def prepare(self):
        """
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos = []
        for combo in itertools.product(*self.all_args):
            self.argument_combos.append(combo)

    def run(self):
        """
        Create tuples of input and output for every possible combination of arguments.
        """
        if not self.argument_combos:
            logging.warning("Please run `prepare` first.")
        self.results = []
        for combo in self.argument_combos:
            self.result.append(
                openai.ChatCompletion.create(
                    model=combo[0],
                    messages=combo[1],
                    temperature=combo[2],
                    top_p=combo[3],
                    n=combo[4],
                    stream=combo[5],
                    stop=combo[6],
                    max_tokens=combo[7],
                    presence_penalty=combo[8],
                    frequency_penalty=combo[9],
                    logit_bias=combo[10],
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
                eval_fn(
                    self.argument_combos[i][1], self._extract_chat_responses(result)
                )
            )

    def visualize(self):
        """
        Creates and shows a table using the results produced.
        """
        if not self.scores:
            logging.warning("Please run `evaluate` first.")
        table = pd.DataFrame(
            {
                "model": [combo[0] for combo in self.argument_combos],
                "messages": [combo[1] for combo in self.argument_combos],
                "response(s)": [
                    self._extract_chat_responses(result) for result in self.results
                ],
                "score": self.scores,
                "other": [combo[2:] for combo in self.argument_combos],
            }
        )
        if is_interactive():
            display(table)
        else:
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))
