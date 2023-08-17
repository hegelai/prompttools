# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List
from prompttools.experiment import Experiment


class ExperimentationHarness:
    r"""
    Base class for experimentation harnesses. This should not be used directly, please use the subclasses instead.
    """

    experiment: Experiment
    PIVOT_COLUMNS: list

    def __init__(self) -> None:
        self.input_pairs_dict = None
        self.experiment = None
        self.runs = 1

    @staticmethod
    def _prepare_arguments(arguments: Dict[str, object]) -> Dict[str, List[object]]:
        return {name: [arg] for name, arg in arguments.items()}

    def prepare(self) -> None:
        r"""
        Prepares the underlying experiment.
        """
        self.experiment.prepare()

    def run(self) -> None:
        r"""
        Runs the underlying experiment.
        """
        self.experiment.run(runs=self.runs)

    def evaluate(self, metric_name: str, eval_fn: Callable, static_eval_fn_kwargs: dict = {}, **eval_fn_kwargs) -> None:
        r"""
        Uses the given eval_fn to evaluate the results of the underlying experiment.
        """
        self.experiment.evaluate(metric_name, eval_fn, static_eval_fn_kwargs, **eval_fn_kwargs)

    # def gather_feedback(self) -> None:
    #     self.experiment.gather_feedback(self.input_pairs_dict, self.PIVOT_COLUMNS)

    def visualize(self, pivot: bool = False) -> None:
        r"""
        Displays a visualization of the experiment results.
        """
        if pivot:
            self.experiment.visualize(pivot_columns=self.PIVOT_COLUMNS, pivot=True)
        else:
            self.experiment.visualize()

    def rank(self, metric_name: str, is_average: bool = False) -> Dict[str, float]:
        r"""
        Scores and ranks the experiment inputs using the pivot columns,
        e.g. prompt templates or system prompts.
        """
        return self.experiment.rank(self.input_pairs_dict, self.PIVOT_COLUMNS, metric_name, is_average)
