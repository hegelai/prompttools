import csv

from prompttools.experiment.openai_chat_experiment import OpenAIChatExperiment
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
        print(human_messages)
        self.system_prompts[system_prompt_file] = system_prompt
        self.human_messages[human_messages_file] = human_messages
        return system_prompt, human_messages

    def _get_harness(self, model_name, system_prompt, human_messages):
        return SystemPromptExperimentationHarness(
            OpenAIChatExperiment, model_name, [system_prompt], human_messages
        )


system_prompt_test_runner = SystemPromptTestRunner()
