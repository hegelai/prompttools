# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Type
from .harness import ExperimentationHarness, Experiment


class SystemPromptExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test various system prompts.

    Args:
        experiment (Type[Experiment]): The experiment that you would like to execute (e.g.
            ``prompttools.experiment.OpenAICompletionExperiment``)
        model_name (str): The name of the model.
        system_prompts (List[str]): A list of system prompts
        human_messages (List[str]): A list of
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``.
    """

    PIVOT_COLUMNS = ["system_prompt", "user_input"]

    def __init__(
        self,
        experiment: Type[Experiment],
        model_name: str,
        system_prompts: List[str],
        human_messages: List[str],
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.experiment_cls_constructor = (experiment,)
        self.model_name = model_name
        self.system_prompts = system_prompts
        self.human_messages = human_messages
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    @staticmethod
    def _create_system_prompt(content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}

    @staticmethod
    def _create_human_message(content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    def prepare(self) -> None:
        r"""
        Creates messages to use for the experiment, and then initializes and prepares the experiment.
        """
        self.input_pairs_dict = {}
        messages_to_try = []
        for system_prompt in self.system_prompts:
            for message in self.human_messages:
                history = [
                    self._create_system_prompt(system_prompt),
                    self._create_human_message(message),
                ]
                messages_to_try.append(history)
                self.input_pairs_dict[str(history)] = (system_prompt, message)
        self.experiment = self.experiment_cls_constructor(
            [self.model_name],
            messages_to_try,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self):
        if not self.experiment:
            self.prepare()
        super().run()
