from typing import Callable, Dict, Optional, Tuple
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

    @staticmethod
    def _is_interactive() -> bool:
        import __main__ as main

        return not hasattr(main, "__file__")

    def _aggregrate_metric(
        self, table, metric_name: str, agg_column, is_average=False
    ) -> pd.DataFrame:
        # TODO: This could be a group by
        prompt_scores = defaultdict(int)
        prompt_counts = defaultdict(int)
        for index, row in table.iterrows():
            prompt_scores[row[agg_column]] += self.scores[metric_name][index]
            prompt_counts[row[agg_column]] += 1
        if is_average:
            prompt_scores[row[agg_column]] /= prompt_counts[row[agg_column]]
        sorted_scores = dict(
            sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_scores

    def _get_human_eval_listener(self, i) -> Callable:
        def listener(change):
            self.scores["feedback"][i] = change["new"]
            if self.use_dialectic_scribe:
                self.completion_fn.add_feedback(
                    self.results[i]["hegel_id"],
                    {"thumbs_up": self.scores["feedback"][i]},
                )

        return listener

    def _get_feedback_submission_listener(self, table, pivot_columns) -> Callable:
        def on_click(b):
            sorted_scores = self._aggregrate_metric(table, "feedback", pivot_columns[0])
            data = {
                pivot_columns[0]: sorted_scores.keys(),
                "feedback": sorted_scores.values(),
            }
            df = pd.DataFrame(data)
            display.display(df)

        return on_click

    def _create_args_dict(self, args, tagname, input_pairs) -> Dict[str, object]:
        args = {self.PARAMETER_NAMES[i]: arg for i, arg in enumerate(args)}
        if self.use_dialectic_scribe:
            args["hegel_tags"] = {tagname: input_pairs[args[1]]}
        return {name: arg for name, arg in args.items() if arg and arg != float("inf")}

    def prepare(self) -> None:
        """
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos = []
        for combo in itertools.product(*self.all_args):
            self.argument_combos.append(combo)

    def run(
        self,
        tagname: str,
        input_pairs: Optional[Dict[str, Tuple[str, Dict[str, str]]]] = None,
    ) -> None:
        """
        Create tuples of input and output for every possible combination of arguments.
        """
        if not self.argument_combos:
            logging.warning("Please run `prepare` first.")
            return
        for combo in self.argument_combos:
            self.queue.enqueue(
                self.completion_fn, self._create_args_dict(combo, tagname, input_pairs)
            )
        self.results = self.queue.results()
        self.scores["latency"] = self.queue.latencies()

    def evaluate(
        self,
        metric_name: str,
        eval_fn: Callable,
        input_pairs: Optional[Dict[str, Tuple[str, Dict[str, str]]]] = None,
    ) -> None:
        """
        Using the given evaluation function, all input/response pairs are evaluated.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        if metric_name in self.scores:
            return
        for i, result in enumerate(self.results):
            # Pass the messages and results into the eval function
            score = eval_fn(
                input_pairs[self.argument_combos[i][1]]
                if input_pairs
                else self.argument_combos[i][1],
                result,
                {
                    name: self.scores[name][i]
                    for name in self.scores.keys()
                    if name is not metric_name
                },
            )
            self.scores[metric_name].append(score)
            if self.use_dialectic_scribe:
                self.completion_fn.add_feedback(
                    self.results[i]["hegel_id"], {metric_name: score}
                )

    def get_table(self, pivot_data, pivot_columns, pivot) -> pd.DataFrame:
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = {
            "messages": [combo[1] for combo in self.argument_combos],
            "response(s)": [self._extract_responses(result) for result in self.results],
            "latency": self.scores["latency"],
        }
        # Add scores for each eval fn, including feedback
        for metric_name, evals in self.scores.items():
            data[metric_name] = evals
        # Add other args as cols if there was more than 1 input
        for i, args in enumerate([self.all_args[0]] + self.all_args[2:]):
            if len(args) > 1:
                data[self.PARAMETER_NAMES[i]] = [
                    combo[i] for combo in self.argument_combos
                ]
        df = None
        if pivot_data:
            data[pivot_columns[0]] = [
                str(pivot_data[str(combo[1])][0]) for combo in self.argument_combos
            ]
            data[pivot_columns[1]] = [
                str(pivot_data[str(combo[1])][1]) for combo in self.argument_combos
            ]
            df = pd.DataFrame(data)
            if pivot:
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

    def gather_feedback(self, pivot_data, pivot_columns) -> None:
        """
        This method creates a table to gather human feedback from a notebook interface.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        self.scores["feedback"] = [1] * len(self.results)
        table = self.get_table(pivot_data, pivot_columns, pivot=False)
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
                layout={"width": "50px"},
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
            layout=widgets.Layout(grid_template_columns="repeat(4, 230px)"),
        )
        display.display(grid)

    def visualize(self, pivot_data=None, pivot_columns=None) -> None:
        """
        Creates and shows a table using the results produced.
        """
        if not self.results:
            logging.warning("Please run `run` first.")
            return
        table = self.get_table(pivot_data, pivot_columns, pivot=True)
        if self._is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))

    def rank(self, pivot_data, pivot_columns, metric_name, is_average):
        """
        Using pivot data, groups the data by the first pivot column to
        get scores, and sorts descending. For example, using pivot data of
        (prompt_template, user_input), a metric of latency, and is_average=True,
        we rank prompt templates by their average latency in the test set.
        """
        if metric_name not in self.scores:
            logging.warning(
                "Can't find " + metric_name + " in scores. Did you run `evaluate`?"
            )
            return
        table = self.get_table(pivot_data, pivot_columns, pivot=False)
        sorted_scores = self._aggregrate_metric(
            table, metric_name, pivot_columns[0], is_average
        )
        return sorted_scores
