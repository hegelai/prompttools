# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional
import openai


from prompttools.mock.mock import mock_chat_completion_fn
from .experiment import Experiment


class OpenAIChatExperiment(Experiment):
    r"""
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.
    """

    PARAMETER_NAMES = ["model", "prompt"]

    def __init__(
        self,
        model: List[str],
        prompt: List[str],
        **kwargs: Dict[str, object],
    ):
        self.completion_fn = openai.ChatCompletion.create
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_chat_completion_fn
        self.all_args = []
        self.all_args.append(model)
        self.all_args.append(prompt)
        for k, v in kwargs.items():
            self.PARAMETER_NAMES.append(k)
            self.all_args.append(v)
        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]]
