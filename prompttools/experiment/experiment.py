# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import itertools
import logging
from IPython import display
from tabulate import tabulate
import pandas as pd

from prompttools.requests.request_queue import RequestQueue
from prompttools.experiment.widgets.feedback import FeedbackWidgetProvider
from prompttools.experiment.widgets.comparison import ComparisonWidgetProvider
from prompttools.experiment.widgets.utility import is_interactive

pd.set_option("display.max_colwidth", 0)


class Experiment:

    r"""
    Base class for experiment. This should not be used directly, please use the subclasses instead.
    """

    completion_fn: Callable
    use_scribe: bool
    PARAMETER_NAMES: tuple
    all_args: list

    def __init__(self):
        self.queue = RequestQueue()
        self.argument_combos = []
        self.results = []
        self.scores = defaultdict(list)
        self.feedback_widget_provider = FeedbackWidgetProvider(
            self.completion_fn, self._aggregate_metric, self._get_human_eval_listener
        )
        self.comparison_widget_provider = ComparisonWidgetProvider(
            self.completion_fn,
            self._aggregate_comparison,
            self._get_comparison_listener,
        )

    def _get_human_eval_listener(self, i: int) -> Callable:
        def listener(change):
            self.scores["feedback"][i] = change["new"]
            if self.use_scribe:
                self.completion_fn.add_feedback(
                    self.results[i]["hegel_id"],
                    {"thumbs_up": self.scores["feedback"][i]},
                )

        return listener

    def _get_comparison_listener(self, index: int) -> Callable:
        # TODO: Map this index to the original index of the primary model
        # so we can submit feedback for the row
        def listener(change):
            new_index = self.comparison_index_translation(index)
            self.scores["comparison"][new_index] = change["new"]
            if self.use_scribe:
                self.completion_fn.add_feedback(
                    self.results[new_index]["hegel_id"],
                    {"thumbs_up": self.scores["comparison"][new_index]},
                )

        return listener

    def _aggregate_comparison(
        self,
        table: pd.DataFrame,
        agg_column: int = 0,
        is_average: bool = False,
    ) -> Dict[str, int]:
        # TODO: This could be a group by
        prompt_scores = defaultdict(int)
        prompt_counts = defaultdict(int)
        for index, row in enumerate(table.iterrows()):
            key = str(row[agg_column])
            new_index = self.comparison_index_translation(index)
            prompt_scores[key] += self.scores["comparison"][new_index]
            prompt_counts[key] += 1
        if is_average:
            for k, v in prompt_scores.items():
                prompt_scores[k] = v / prompt_counts[k]
        sorted_scores = dict(
            sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_scores

    def _aggregate_metric(
        self,
        table: pd.DataFrame,
        metric_name: str,
        agg_column: str,
        is_average: bool = False,
    ) -> Dict[str, int]:
        # TODO: This could be a group by

        prompt_scores = defaultdict(int)
        prompt_counts = defaultdict(int)
        for index, row in table.iterrows():
            key = str(row[1][agg_column])
            prompt_scores[key] += self.scores[metric_name][index]
            prompt_counts[key] += 1
        if is_average:
            for k, v in prompt_scores.items():
                prompt_scores[k] = v / prompt_counts[k]
        sorted_scores = dict(
            sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_scores

    def _create_args_dict(
        self, args: Dict[str, object], tagname: str, input_pairs: Dict[str, object]
    ) -> Dict[str, object]:
        args = {self.PARAMETER_NAMES[i]: arg for i, arg in enumerate(args)}
        if self.use_scribe:
            args["hegel_tags"] = {tagname: input_pairs[args[1]]}
        return {name: arg for name, arg in args.items() if arg and arg != float("inf")}

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos = []
        for combo in itertools.product(*self.all_args):
            self.argument_combos.append(combo)

    def run(
        self,
        tagname: Optional[str] = "",
        input_pairs: Optional[Dict[str, Tuple[str, Dict[str, str]]]] = None,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        """
        if not self.argument_combos:
            logging.info('Preparing first...')
            self.prepare()
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
            logging.info('Running first...')
            self.run()
        if metric_name in self.scores:
            logging.warning(metric_name + " is already present, skipping.")
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
            if self.use_scribe:
                self.completion_fn.add_feedback(
                    self.results[i]["hegel_id"], {metric_name: score}
                )

    def get_table(
        self, pivot_data: Dict[str, object], pivot_columns: List[str], pivot: bool
    ) -> pd.DataFrame:
        """
        This method creates a table of the experiment data. It can also be used
        to create a pivot table, or a table for gathering human feedback.
        """
        data = {
            "messages": [str(combo[1]) for combo in self.argument_combos],
            "response(s)": [self._extract_responses(result) for result in self.results],
            "latency": self.scores["latency"],
        }
        # Add scores for each eval fn, including feedback
        for metric_name, evals in self.scores.items():
            if metric_name != "comparison":
                data[metric_name] = evals
        # Add other args as cols if there was more than 1 input
        for i, args in enumerate(self.all_args):
            if len(args) > 1:
                data[self.PARAMETER_NAMES[i]] = [
                    combo[i] for combo in self.argument_combos
                ]
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
        return df

    def gather_feedback(
        self, pivot_data: Dict[str, object], pivot_columns: List[str]
    ) -> None:
        """
        This method creates a table to gather human feedback from a notebook interface.
        """
        if not self.results:
            logging.info('Running first...')
            self.run()
        if not is_interactive():
            logging.warning("This method only works in notebooks.")
            return
        self.scores["feedback"] = [1] * len(self.results)
        table = self.get_table(pivot_data, pivot_columns, pivot=False)
        self.feedback_widget_provider.set_pivot_columns(pivot_columns)
        items = self.feedback_widget_provider.get_header_widgets()
        for row in table.iterrows():
            items += self.feedback_widget_provider.get_row_widgets(*row)
        items += self.feedback_widget_provider.get_footer_widgets(table)
        self.feedback_widget_provider.display(items)

    def compare(self, primary_model: str, pivot_columns: List[str]) -> None:
        """
        This method creates a table to gather human feedback from a notebook interface.
        """
        if not self.results:
            logging.info('Running first...')
            self.run()
        if not is_interactive():
            logging.warning("This method only works in notebooks.")
            return
        table = self.get_table(pivot_data={}, pivot_columns=pivot_columns, pivot=True)
        self.scores["comparison"] = [1] * len(table)
        self.comparison_index_translation = lambda i: i * len(table.columns)
        self.comparison_widget_provider.set_models(table.columns)
        items = self.comparison_widget_provider.get_header_widgets()
        for index, row in enumerate(table.iterrows()):
            items += self.comparison_widget_provider.get_row_widgets(index, row[1])
        items += self.comparison_widget_provider.get_footer_widgets(table)
        self.comparison_widget_provider.display(items)

    def visualize(
        self,
        pivot_data: Optional[Dict[str, object]] = None,
        pivot_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Creates and shows a table using the results produced.
        """
        if not self.results:
            logging.info('Running first...')
            self.run()
        table = self.get_table(
            pivot_data, pivot_columns, pivot=pivot_columns is not None
        )
        if is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))

    def rank(
        self,
        pivot_data: Dict[str, object],
        pivot_columns: List[str],
        metric_name: str,
        is_average: bool,
    ) -> Dict[str, int]:
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
        sorted_scores = self._aggregate_metric(
            table, metric_name, pivot_columns[0], is_average
        )
        return sorted_scores

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        raise NotImplementedError("This should be implemented by a subclass of `Experiment`.")
