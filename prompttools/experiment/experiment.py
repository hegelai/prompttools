from typing import Callable, Dict
from collections import defaultdict
import itertools
import logging
from IPython import display
from tabulate import tabulate
import pandas as pd
import ipywidgets as widgets

from prompttools.requests.request_queue import RequestQueue

pd.set_option("display.max_colwidth", 0)


class Experiment:
    def __init__(self):
        self.queue = RequestQueue()
        self.argument_combos = []
        self.results = []
        self.scores = defaultdict(list)
        self.human_evals = []

    @staticmethod
    def _is_interactive():
        import __main__ as main

        return not hasattr(main, "__file__")

    def _get_human_eval_listener(self, i):
        def listener(change):
            self.human_evals[i] = change["new"]

        return listener

    def _get_feedback_submission_listener(self, table, pivot_columns):
        def on_click(b):
            prompt_scores = defaultdict(int)
            for index, row in table.iterrows():
                prompt_scores[row[pivot_columns[0]]] += self.human_evals[index]
            sorted_scores = dict(
                sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)
            )
            data = {
                pivot_columns[0]: sorted_scores.keys(),
                "feedback_score": sorted_scores.values(),
            }
            df = pd.DataFrame(data)
            display.display(df)

        return on_click

    # TODO: Agg function for eval scores

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

    def evaluate(self, metric_name: str, eval_fn: Callable):
        """
        Using the given evaluation function, all input/response pairs are evaluated.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        for i, result in enumerate(self.results):
            # Pass the messages and results into the eval function
            self.scores[metric_name].append(
                eval_fn(self.argument_combos[i][1], result)
            )

    def get_table(self, pivot_data, pivot_columns, gather_feedback):
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = {
            "messages": [combo[1] for combo in self.argument_combos],
            "response(s)": [self._extract_responses(result) for result in self.results],
            "latency": self.latencies,
        }
        # Add scores for each eval fn
        for metric_name, evals in self.scores.items():
            data[metric_name] = evals
        if self.human_evals:
            data["feedback"] = self.human_evals
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

    def gather_feedback(self, pivot_data, pivot_columns):
        """
        This method creates a table to gather human feedback from a notebook interface.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        self.human_evals = [1] * len(self.results)
        table = self.get_table(pivot_data, pivot_columns, gather_feedback=True)
        items = [
            widgets.Label(pivot_columns[0]),
            widgets.Label(pivot_columns[1]),
            widgets.Label("response(s)"),
            widgets.Label("Feedback"),
        ]
        for index, row in table.iterrows():
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

            feedback_dropdown = widgets.Dropdown(
                options=[("\U0001F44D", 1), ("\U0001F44E", 0)],
                value=1,
                description="Feedback:",
            )
            feedback_dropdown.observe(
                self._get_human_eval_listener(index), names="value"
            )
            items += [feedback_dropdown]
        submit_button = widgets.Button(
            description="Submit",
            disabled=False,
            button_style="success",
            tooltip="Submit",
        )
        submit_button.on_click(
            self._get_feedback_submission_listener(table, pivot_columns)
        )
        items += [
            widgets.Label(""),
            widgets.Label(""),
            widgets.Label(""),
            submit_button,
        ]
        grid = widgets.GridBox(
            items,
            layout=widgets.Layout(grid_template_columns="repeat(4, 250px)"),
        )
        display.display(grid)

    def visualize(self, pivot_data=None, pivot_columns=None):
        """
        Creates and shows a table using the results produced.
        """
        if not self.scores:
            logging.warning("Please run `evaluate` first.")
            return
        table = self.get_table(pivot_data, pivot_columns, gather_feedback=False)
        if self._is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))
