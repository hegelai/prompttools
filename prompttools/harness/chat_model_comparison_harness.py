# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from prompttools.harness.harness import ExperimentationHarness
from prompttools.experiment.openai_chat_experiment import OpenAIChatExperiment


class ChatModelComparisonHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for comparing chat models.
    """

    PIVOT_COLUMNS = ["model", "messages"]

    def __init__(
        self,
        model_names: List[str],
        chat_histories: List[List[Dict[str, str]]],
        runs: Optional[int] = 1,
        use_scribe: Optional[bool] = False,
        scribe_name: Optional[str] = "Chat Model Comparison",
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.experiment_classname = OpenAIChatExperiment
        self.model_names = model_names
        self.chat_histories = chat_histories
        self.runs = runs
        self.use_scribe = use_scribe
        self.scribe_name = scribe_name
        self.model_arguments = model_arguments
        super().__init__()

    def prepare(self) -> None:
        """
        Initializes and prepares the experiment.
        """
        histories_to_try = []
        for history in self.chat_histories:
            histories_to_try.extend([history] * self.runs)

        self.experiment = self.experiment_classname(
            self.model_names,
            histories_to_try,
            self.use_scribe,
            self.scribe_name,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def compare(self):
        self.experiment.compare(self.model_names[0], self.PIVOT_COLUMNS)
