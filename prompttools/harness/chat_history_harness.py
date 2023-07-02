from typing import Dict, List, Optional
from prompttools.harness.harness import ExperimentationHarness


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
