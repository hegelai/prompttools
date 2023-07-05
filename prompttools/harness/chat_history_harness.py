# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from prompttools.harness.harness import ExperimentationHarness
from prompttools.experiment.openai_chat_experiment import OpenAIChatExperiment


class ChatHistoryExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for chat histories.
    """

    def __init__(
        self,
        model_name: str,
        chat_histories: List[List[Dict[str, str]]],
        use_scribe: Optional[bool] = False,
        scribe_name: Optional[str] = "Chat History Experiment",
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.experiment_classname = OpenAIChatExperiment
        self.model_name = model_name
        self.chat_histories = chat_histories
        self.use_scribe = use_scribe
        self.scribe_name = scribe_name
        self.model_arguments = model_arguments

    def prepare(self) -> None:
        r"""
        Initializes and prepares the experiment.
        """
        self.experiment = self.experiment_classname(
            [self.model_name],
            self.chat_histories,
            self.use_scribe,
            self.scribe_name,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()
