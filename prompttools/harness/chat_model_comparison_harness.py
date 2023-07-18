# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from .harness import ExperimentationHarness
from prompttools.experiment import OpenAIChatExperiment


class ChatModelComparisonHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for comparing chat models.
    Multi-model version of ``ChatHistoryExperimentationHarness``.

    Args:
        model_names (List[str]): The names of the models that you would like to compare
        chat_histories (List[List[Dict[str, str]]]): A list of chat histories that will be fed into the models.
        runs (int): Number of runs to execute. Defaults to ``1``.
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``.
    """

    PIVOT_COLUMNS = ["model", "messages"]

    def __init__(
        self,
        model_names: List[str],
        chat_histories: List[List[Dict[str, str]]],
        runs: int = 1,
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.experiment_cls_constructor = OpenAIChatExperiment
        self.model_names = model_names
        self.chat_histories = chat_histories
        self.runs = runs
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    def prepare(self) -> None:
        """
        Initializes and prepares the experiment.
        """
        self.experiment = self.experiment_cls_constructor(
            self.model_names,
            self.chat_histories,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self):
        if not self.experiment:
            self.prepare()
        super().run()

    def compare(self):
        self.experiment.compare(self.model_names[0], self.PIVOT_COLUMNS)
