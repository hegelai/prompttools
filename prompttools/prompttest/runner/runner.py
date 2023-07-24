# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Type
import logging

from prompttools.prompttest.threshold_type import ThresholdType
from prompttools.prompttest.error.failure import log_failure
from prompttools.experiment import Experiment
from prompttools.prompttest.error.failure import PromptTestSetupException


class PromptTestRunner:
    r"""
    Base class for prompt test runners. Please use the subclass instead.s
    """

    def __init__(self):
        self.ran = defaultdict(bool)
        self.experiments = dict()

    def run(self, *args, **kwargs) -> str:
        r"""
        Runs the test if it has not already been run.
        """
        key = str(args)
        if self.ran[key]:
            return key
        self.experiments[key] = self._get_experiment(*args, **kwargs)
        self.experiments[key].run()
        self.ran[key] = True
        return key

    def evaluate(
        self,
        key: str,
        metric_name: str,
        eval_fn: Callable,
        expected: Optional[str] = None,
    ) -> None:
        r"""
        Evaluates the test results using the given ``eval_fn``.
        """
        self.experiments[key].evaluate(metric_name, eval_fn, expected=expected)

    def visualize(self, key: str) -> None:
        r"""
        Evaluates the test results using the given ``eval_fn``.
        """
        self.experiments[key].visualize()

    def scores(self, key):
        r"""
        Returns the scores for the underlying experiment at the
        given key.
        """
        return self.experiments[key].scores

    @staticmethod
    def _get_experiment(
        experiment: Type[Experiment],
        model_name: str,
        prompts: List[str],
        model_args: Dict[str, object],
    ) -> Experiment:
        return experiment([model_name], prompts, **{k: [v] for k, v in model_args})


prompt_test_runner = PromptTestRunner()


def run_prompttest(
    metric_name: str,
    eval_fn: Callable,
    threshold: float,
    threshold_type: ThresholdType,
    prompts: List[str],
    results: List[str],
    expected: Optional[List[str]],
) -> int:
    """
    Runs the prompt test evaluation.
    """
    scores = []
    for i, result in enumerate(results):
        if expected:
            score = eval_fn(prompts[i], result, metadata={}, expected=expected[i])
        else:
            score = eval_fn(prompts[i], result, metadata={})
        scores.append(score)
    if not scores:
        logging.error("Something went wrong during testing. Make sure your API keys are set correctly.")
        raise PromptTestSetupException
    for score in scores:
        if not (score <= threshold if threshold_type == ThresholdType.MAXIMUM else score >= threshold):
            log_failure(metric_name, threshold, score, threshold_type)
            return 1
    return 0
