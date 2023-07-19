# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Tuple, Type
import csv
import logging

from prompttools.prompttest.threshold_type import ThresholdType
from prompttools.prompttest.error.failure import log_failure
from prompttools.experiment import Experiment
from prompttools.harness import SystemPromptExperimentationHarness
from prompttools.prompttest.runner.runner import PromptTestRunner
from prompttools.prompttest.error.failure import PromptTestSetupException


class SystemPromptTestRunner(PromptTestRunner):
    r"""
    A prompt test runner for tests based on system prompts.
    """

    def __init__(self):
        self.system_prompts = {}
        self.human_messages = {}
        super().__init__()

    def read(
        self, system_prompt_file: str, human_messages_file: str
    ) -> Tuple[str, List[List[str]]]:
        r"""
        Reads data from files and parses it into a system prompt and human messages.
        """
        if (
            system_prompt_file in self.system_prompts
            and human_messages_file in self.human_messages
        ):
            return (
                self.system_prompts[system_prompt_file],
                self.human_messages[human_messages_file],
            )
        with open(system_prompt_file, "r") as f:
            system_prompt = f.read()
        with open(human_messages_file) as f:
            human_messages = list(csv.reader(f))
        self.system_prompts[system_prompt_file] = system_prompt
        self.human_messages[human_messages_file] = human_messages
        return system_prompt, human_messages

    @staticmethod
    def _get_harness(
        experiment: Type[Experiment],
        model_name: str,
        system_prompt: str,
        human_messages: List[str],
        model_args: Dict[str, object],
    ) -> SystemPromptExperimentationHarness:
        return SystemPromptExperimentationHarness(
            experiment,
            model_name,
            [system_prompt],
            human_messages,
            model_arguments=model_args,
        )


system_prompt_test_runner = SystemPromptTestRunner()


def run_system_prompt_test(
    experiment: Type[Experiment],
    model_name: str,
    metric_name: str,
    eval_fn: Callable,
    threshold: float,
    threshold_type: ThresholdType,
    is_average: bool,
    system_prompt: str,
    human_messages: List[str],
    use_input_pairs: bool,
    model_args: Dict[str, object],
) -> int:
    r"""
    Runs the prompt test.
    """
    key = system_prompt_test_runner.run(
        experiment, model_name, system_prompt, human_messages, model_args
    )
    system_prompt_test_runner.evaluate(key, metric_name, eval_fn, use_input_pairs)
    scored_template = system_prompt_test_runner.rank(key, metric_name, is_average)
    if not scored_template:
        logging.error(
            "Something went wrong during testing. Make sure your API keys are set correctly."
        )
        raise PromptTestSetupException
    if (
        scored_template[system_prompt] < threshold
        and threshold_type is ThresholdType.MINIMUM
    ):
        log_failure(
            metric_name,
            threshold,
            actual=scored_template[system_prompt],
            threshold_type=threshold_type,
        )
        return 1
    if (
        scored_template[system_prompt] > threshold
        and threshold_type is ThresholdType.MAXIMUM
    ):
        log_failure(
            metric_name,
            threshold,
            actual=scored_template[system_prompt],
            threshold_type=threshold_type,
        )
        return 1
    return 0


def run_system_prompt_test_from_files(
    experiment: Type[Experiment],
    model_name: str,
    metric_name: str,
    eval_fn: Callable,
    threshold: float,
    threshold_type: ThresholdType,
    is_average: bool,
    system_prompt_file: str,
    human_messages_file: str,
    use_input_pairs: bool,
    model_args: Dict[str, object],
) -> int:
    r"""
    Reads data in from files and runs the prompt test.
    """
    system_prompt, human_messages = system_prompt_test_runner.read(
        system_prompt_file, human_messages_file
    )
    return run_system_prompt_test(
        experiment,
        model_name,
        metric_name,
        eval_fn,
        threshold,
        threshold_type,
        is_average,
        system_prompt,
        human_messages,  # TODO: The type of this may be incorrect
        use_input_pairs,
        model_args,
    )
