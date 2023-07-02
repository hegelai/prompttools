from typing import Callable, Dict
import itertools
import logging
from IPython import display
from tabulate import tabulate
import pandas as pd


class Experiment:
    def __init__(self):
        self.argument_combos = None
        self.results = None
        self.scores = None

    @staticmethod
    def _is_interactive():
        import __main__ as main

        return not hasattr(main, "__file__")

    def _create_args_dict(self, args) -> Dict[str, object]:
        args = {self.PARAMETER_NAMES[i]: arg for i, arg in enumerate(args)}
        return {name: arg for name, arg in args.items() if arg and arg != float("inf")}

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
            return
        for combo in self.argument_combos:
            self.queue.enqueue(self.completion_fn, self._create_args_dict(combo))
        self.results = self.queue.results()
        self.latencies = self.queue.latencies()

    def evaluate(self, eval_fn: Callable):
        """
        Using the given evaluation function, all input/response pairs are evaluated.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        self.scores = []
        for i, result in enumerate(self.results):
            # Pass the messages and results into the eval function
            self.scores.append(
                eval_fn(self.argument_combos[i][1], self._extract_responses(result))
            )

    def get_table(self):
        if not self.scores:
            logging.warning("Please run `evaluate` first.")
            return None
        data = {
            "messages": [combo[1] for combo in self.argument_combos],
            "response(s)": [self._extract_responses(result) for result in self.results],
            "score": self.scores,
            "latencies": self.latencies,
        }
        # Add other args as cols if there was more than 1 input
        for i, args in enumerate([self.all_args[0]] + self.all_args[2:]):
            if len(args) > 1:
                data[self.PARAMETER_NAMES[i]] = (
                    [combo[i] for combo in self.argument_combos],
                )
        return pd.DataFrame(data)

    def visualize(self):
        """
        Creates and shows a table using the results produced.
        """
        table = self.get_table()
        if table is None:
            return
        if self._is_interactive():
            display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))
