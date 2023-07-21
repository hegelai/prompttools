# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Callable, Dict, List, Tuple


class PromptTestRunner:
    r"""
    Base class for prompt test runners. Please use the subclass instead.s
    """

    def __init__(self):
        self.ran = defaultdict(bool)
        self.harnesses = dict()

    def run(self, *args, **kwargs) -> str:
        r"""
        Runs the test if it has not already been run.
        """
        key = str(args)
        if self.ran[key]:
            return key
        self.harnesses[key] = self._get_harness(*args, **kwargs)
        self.harnesses[key].run()
        self.ran[key] = True
        return key

    def evaluate(
        self,
        key: str,
        metric_name: str,
        eval_fn: Callable,
        use_input_pairs: Dict[str, Tuple[str, Dict[str, str]]],
    ) -> None:
        r"""
        Evaluates the test results.
        """
        self.harnesses[key].evaluate(metric_name, eval_fn, use_input_pairs)

    def rank(self, key: str, metric_name: str, is_average: bool) -> Dict[str, float]:
        r"""
        Uses evaluations to "rank" the template or system prompt.
        This creates an overall score (sum or average) across all
        test inputs, which can be compared against a threshold.
        """
        return self.harnesses[key].rank(metric_name, is_average)

    @staticmethod
    def _get_harness(
        experiment,
        model_name: str,
        prompt_template: str,
        user_inputs: List[Dict[str, str]],
        model_args: Dict[str, object],
    ):
        raise NotImplementedError("This should be implemented by a subclass of `PromptTestRunner`.")
