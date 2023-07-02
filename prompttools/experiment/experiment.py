from typing import Callable, Dict
import itertools
import logging
from IPython import display
from tabulate import tabulate
import pandas as pd
import ipywidgets as widgets

from prompttools.requests.request_queue import RequestQueue


class Experiment:
    def __init__(self):
        self.queue = RequestQueue()
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

    def get_table(self, pivot_data, pivot_columns, gather_feedback):
        if not self.scores:
            logging.warning("Please run `evaluate` first.")
            return None
        data = {
            "messages": [combo[1] for combo in self.argument_combos],
            "response(s)": [self._extract_responses(result) for result in self.results],
            "score": self.scores,
            "latency": self.latencies,
        }
        # Add other args as cols if there was more than 1 input
        for i, args in enumerate([self.all_args[0]] + self.all_args[2:]):
            if len(args) > 1:
                data[self.PARAMETER_NAMES[i]] = [
                    combo[i] for combo in self.argument_combos
                ]
        df = None
        if pivot_data:
            data[pivot_columns[0]] = [
                pivot_data[str(combo[1])][0] for combo in self.argument_combos
            ]
            data[pivot_columns[1]] = [
                pivot_data[str(combo[1])][1] for combo in self.argument_combos
            ]
            df = pd.DataFrame(data)
            if not gather_feedback:
                df = pd.pivot_table(
                    df,
                    values="response(s)",
                    index=[pivot_columns[1]],
                    columns=[pivot_columns[0]],
                    aggfunc=lambda x: x.iloc[0],
                )
        else:
            df = pd.DataFrame(data)
        return df

    def visualize(self, pivot_data=None, pivot_columns=None, gather_feedback=False):
        """
        Creates and shows a table using the results produced.
        """
        table = self.get_table(pivot_data, pivot_columns, gather_feedback)
        if table is None:
            return

        if self._is_interactive():
            if gather_feedback:
                items = [
                    widgets.Label(pivot_columns[0]),
                    widgets.Label(pivot_columns[1]),
                    widgets.Label("response(s)"),
                    widgets.Label("Feedback"),
                ]
                for _, row in table.iterrows():
                    items += [
                        widgets.HTML(
                            value="<style>p{word-wrap: break-word}</style> <p>"
                            + row[pivot_columns[0]]
                            + " </p>"
                        )
                    ]
                    items += [
                        widgets.HTML(
                            value="<style>p{word-wrap: break-word}</style> <p>"
                            + row[pivot_columns[1]]
                            + " </p>"
                        )
                    ]
                    items += [
                        widgets.HTML(
                            value="<style>p{word-wrap: break-word}</style> <p>"
                            + ", ".join(row["response(s)"])
                            + " </p>"
                        )
                    ]
                    items += [
                        widgets.Dropdown(
                            options=[("\U0001F44D", 1), ("\U0001F44E", 2)],
                            value=1,
                            description="Feedback:",
                        )
                    ]
                grid = widgets.GridBox(
                    items,
                    layout=widgets.Layout(grid_template_columns="repeat(4, 250px)"),
                )
                display.display(grid)
            else:
                display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))
