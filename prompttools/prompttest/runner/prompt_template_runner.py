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
from prompttools.harness import PromptTemplateExperimentationHarness
from prompttools.prompttest.runner.runner import PromptTestRunner
from prompttools.prompttest.error.failure import PromptTestSetupException


class PromptTemplateTestRunner(PromptTestRunner):
    r"""
    A prompt test runner for tests based on prompt templates.
    """

    def __init__(self):
        self.prompt_templates = {}
        self.user_inputs = {}
        super().__init__()

    def read(
        self, prompt_template_file: str, user_input_file: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Reads data from files and parses it into a prompt template and user input.
        """
        if (
            prompt_template_file in self.prompt_templates
            and user_input_file in self.user_inputs
        ):
            return (
                self.prompt_templates[prompt_template_file],
                self.user_inputs[user_input_file],
            )
        prompt_template = ""
        with open(prompt_template_file, "r") as f:
            prompt_template = f.read()
        user_inputs = [{"": ""}]
        with open(user_input_file) as f:
            user_inputs = list(csv.DictReader(f))
        self.prompt_templates[prompt_template_file] = prompt_template
        self.user_inputs[user_input_file] = user_inputs
        return prompt_template, user_inputs

    @staticmethod
    def _get_harness(
        experiment: Type[Experiment],
        model_name: str,
        prompt_template: str,
        user_inputs: List[Dict[str, str]],
        model_args: Dict[str, object],
    ) -> PromptTemplateExperimentationHarness:
        return PromptTemplateExperimentationHarness(
            experiment,
            model_name,
            [prompt_template],
            user_inputs,
            model_arguments=model_args,
        )


prompt_template_test_runner = PromptTemplateTestRunner()


def run_prompt_template_test(
    experiment: Type[Experiment],
    model_name: str,
    metric_name: str,
    eval_fn: Callable,
    threshold: float,
    threshold_type: ThresholdType,
    is_average: bool,
    prompt_template: str,
    user_inputs: List[Dict[str, str]],
    use_input_pairs: bool,
    model_args: Dict[str, object],
) -> int:
    """
    Runs the prompt test.
    """
    key = prompt_template_test_runner.run(
        experiment, model_name, prompt_template, user_inputs, model_args
    )
    prompt_template_test_runner.evaluate(key, metric_name, eval_fn, use_input_pairs)
    scored_template = prompt_template_test_runner.rank(key, metric_name, is_average)
    if not scored_template:
        logging.error(
            "Something went wrong during testing. Make sure your API keys are set correctly."
        )
        raise PromptTestSetupException
    if (
        scored_template[prompt_template] < threshold
        and threshold_type is ThresholdType.MINIMUM
    ):
        log_failure(
            metric_name,
            threshold,
            actual=scored_template[prompt_template],
            threshold_type=threshold_type,
        )
        return 1
    if (
        scored_template[prompt_template] > threshold
        and threshold_type is ThresholdType.MAXIMUM
    ):
        log_failure(
            metric_name,
            threshold,
            actual=scored_template[prompt_template],
            threshold_type=threshold_type,
        )
        return 1
    return 0


def run_prompt_template_test_from_files(
    experiment: Type[Experiment],
    model_name: str,
    metric_name: str,
    eval_fn: Callable,
    threshold: float,
    threshold_type: ThresholdType,
    is_average: bool,
    prompt_template_file: str,
    user_input_file: str,
    use_input_pairs: bool,
    model_args: Dict[str, object],
) -> int:
    """
    Reads data in from files and runs the prompt test.
    """
    prompt_template, user_inputs = prompt_template_test_runner.read(
        prompt_template_file, user_input_file
    )
    return run_prompt_template_test(
        experiment,
        model_name,
        metric_name,
        eval_fn,
        threshold,
        threshold_type,
        is_average,
        prompt_template,
        user_inputs,
        use_input_pairs,
        model_args,
    )
