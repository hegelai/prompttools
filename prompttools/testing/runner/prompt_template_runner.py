import csv

from prompttools.testing.error.failure import log_failure
from prompttools.harness.prompt_template_harness import (
    PromptTemplateExperimentationHarness,
)
from prompttools.testing.runner.runner import PromptTestRunner


class PromptTemplateTestRunner(PromptTestRunner):
    def __init__(self):
        self.prompt_templates = {}
        self.user_inputs = {}
        super().__init__()

    def read(self, prompt_template_file, user_input_file):
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

    def _get_harness(self, model_name, prompt_template, user_inputs):
        return PromptTemplateExperimentationHarness(
            model_name, [prompt_template], user_inputs
        )


prompt_template_test_runner = PromptTemplateTestRunner()


def run_prompt_template_test(
    model_name,
    metric_name,
    eval_fn,
    threshold,
    is_average,
    prompt_template,
    user_inputs,
):
    key = prompt_template_test_runner.run(model_name, prompt_template, user_inputs)
    prompt_template_test_runner.evaluate(key, metric_name, eval_fn)
    scored_template = prompt_template_test_runner.rank(key, metric_name, is_average)
    if scored_template[prompt_template] < threshold:
        log_failure(metric_name, threshold, actual=scored_template[prompt_template])
        return 1
    return 0


def run_prompt_template_test_from_files(
    model_name,
    metric_name,
    eval_fn,
    threshold,
    is_average,
    prompt_template_file,
    user_input_file,
):
    prompt_template, user_inputs = prompt_template_test_runner.read(
        prompt_template_file, user_input_file
    )
    return run_prompt_template_test(
        model_name,
        metric_name,
        eval_fn,
        threshold,
        is_average,
        prompt_template,
        user_inputs,
    )
