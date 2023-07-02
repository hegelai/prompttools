from typing import Dict, List, Optional


# TODO: Create one that accepts a list of prompt templates and inputs
class ExperimentationHarness:
    @staticmethod
    def _prepare_arguments(arguments):
        return {name: [arg] for name, arg in arguments}

    def prepare(self):
        self.experiment.prepare()

    def run(self):
        self.experiment.run()

    def evaluate(self, eval_fn):
        self.experiment.evaluate(eval_fn)


class SystemPromptExperimentationHarness(ExperimentationHarness):
    def __init__(
        self,
        experiment_classname,
        model_name: str,
        system_prompts: List[str],
        human_messages: List[str],
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.experiment_classname = experiment_classname
        self.model_name = model_name
        self.model_arguments = model_arguments
        self.system_prompts = system_prompts
        self.human_messages = human_messages

    @staticmethod
    def _create_system_prompt(content):
        return ({"role": "system", "content": content},)

    @staticmethod
    def _create_human_message(content):
        return ({"role": "user", "content": content},)

    @staticmethod
    def _prepare_arguments(arguments):
        return {name: [arg] for name, arg in arguments}

    def prepare(self):
        messages_to_try = []
        for system_prompt in self.system_prompts:
            for message in self.human_messages:
                messages_to_try.append(
                    [
                        self._create_system_prompt(system_prompt),
                        self._create_human_message(message),
                    ]
                )
        self.experiment = self.experiment_classname(
            [self.model_name],
            messages_to_try,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()


class ChatHistoryExperimentationHarness(ExperimentationHarness):
    def __init__(
        self,
        experiment_classname,
        model_name: str,
        chat_histories: List[List[Dict[str, str]]],
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.experiment_classname = experiment_classname
        self.model_name = model_name
        self.model_arguments = model_arguments
        self.chat_histories = chat_histories

    @staticmethod
    def _prepare_arguments(arguments):
        return {name: [arg] for name, arg in arguments}

    def prepare(self):
        self.experiment = self.experiment_classname(
            [self.model_name],
            self.chat_histories,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()
