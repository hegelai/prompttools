# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from .harness import ExperimentationHarness
from prompttools.experiment import OpenAIChatExperiment


class ChatHistoryExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for compare multiple chat histories.

    Args:
        model_name (str): The name of the model.
        chat_histories (List[List[Dict[str, str]]]): A list of chat histories that will be fed into the model.
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``.
    """

    def __init__(
        self,
        model_name: str,
        chat_histories: List[List[Dict[str, str]]],
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.experiment_cls_constructor = OpenAIChatExperiment
        self.model_name = model_name
        self.chat_histories = chat_histories
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    def prepare(self) -> None:
        r"""
        Initializes and prepares the experiment.
        """
        self.experiment = self.experiment_cls_constructor(
            [self.model_name],
            self.chat_histories,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self):
        if not self.experiment:
            self.prepare()
        super().run()
