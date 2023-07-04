import csv

from prompttools.testing.threshold_type import ThresholdType
from prompttools.testing.error.failure import log_failure
from prompttools.harness.system_prompt_harness import SystemPromptExperimentationHarness
from prompttools.testing.runner.runner import PromptTestRunner


class SystemPromptTestRunner(PromptTestRunner):
    def __init__(self):
        self.system_prompts = {}
        self.human_messages = {}
        super().__init__()

    def read(self, system_prompt_file, human_messages_file):
        if (
            system_prompt_file in self.system_prompts
            and human_messages_file in self.human_messages
        ):
            return (
                self.system_prompts[system_prompt_file],
                self.human_messages[human_messages_file],
            )
        system_prompt = ""
        with open(system_prompt_file, "r") as f:
            system_prompt = f.read()
        human_messages = []
        with open(human_messages_file) as f:
            human_messages = list(csv.reader(f))
        self.system_prompts[system_prompt_file] = system_prompt
        self.human_messages[human_messages_file] = human_messages
        return system_prompt, human_messages

    def _get_harness(self, model_name, system_prompt, human_messages):
        return SystemPromptExperimentationHarness(
            model_name, [system_prompt], human_messages
        )


system_prompt_test_runner = SystemPromptTestRunner()


def run_system_prompt_test(
    model_name,
    metric_name,
    eval_fn,
    threshold,
    threshold_type,
    is_average,
    system_prompt,
    human_messages,
    use_input_pairs,
):
    key = system_prompt_test_runner.run(model_name, system_prompt, human_messages)
    system_prompt_test_runner.evaluate(key, metric_name, eval_fn, use_input_pairs)
    scored_template = system_prompt_test_runner.rank(key, metric_name, is_average)
    if scored_template[system_prompt] < threshold and threshold_type is ThresholdType.MINIMUM:
        log_failure(metric_name, threshold, actual=scored_template[system_prompt], threshold_type=threshold_type)
        return 1
    if scored_template[system_prompt] > threshold and threshold_type is ThresholdType.MAXIMUM:
        log_failure(metric_name, threshold, actual=scored_template[system_prompt], threshold_type=threshold_type)
        return 1
    return 0


def run_system_prompt_test_from_files(
    model_name,
    metric_name,
    eval_fn,
    threshold,
    threshold_type,
    is_average,
    system_prompt_file,
    human_messages_file,
    use_input_pairs,
):
    system_prompt, human_messages = system_prompt_test_runner.read(
        system_prompt_file, human_messages_file
    )
    return run_system_prompt_test(
        model_name,
        metric_name,
        eval_fn,
        threshold,
        threshold_type,
        is_average,
        system_prompt,
        human_messages,
        use_input_pairs,
    )
