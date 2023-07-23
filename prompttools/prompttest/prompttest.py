# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Type
from functools import wraps
import logging

from prompttools.experiment import Experiment
from .threshold_type import ThresholdType
from .error.failure import PromptTestSetupException
from .runner.runner import run_prompttest

TESTS_TO_RUN = []


def prompttest(
    experiment: Type[Experiment],
    model_name: str,
    metric_name: str,
    eval_fn: Callable,
    threshold: float = 1.0,
    threshold_type: ThresholdType = ThresholdType.MAXIMUM,
    expected: Optional[str] = None,
    model_arguments: Dict[str, object] = {},
):
    r"""
    Creates a decorator for prompt tests, which can annotate evaluation functions.
    This enables developers to create a prompt test suite from their evaluations.
    """

    def prompttest_decorator(prompt_provider: Callable):
        @wraps(prompt_provider)
        def runs_test():
            prompts = prompt_provider()
            return run_prompttest(
                experiment,
                model_name,
                metric_name,
                eval_fn,
                threshold,
                threshold_type,
                prompts,
                model_arguments=model_arguments,
                expected=expected,
            )

        TESTS_TO_RUN.append(runs_test)
        return runs_test

    return prompttest_decorator


def main():
    logging.getLogger().setLevel(logging.WARNING)
    print("Running " + str(len(TESTS_TO_RUN)) + " test(s)")
    failures = int(sum([test() for test in TESTS_TO_RUN]))
    if failures == 0:
        print("All " + str(len(TESTS_TO_RUN)) + " test(s) passed!")
        exit(0)
    else:
        print("Tests failed: " + str(failures))
        exit(1)
