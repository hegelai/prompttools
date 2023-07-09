# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional
import openai

import logging

from prompttools.mock.mock import mock_chat_completion_fn
from prompttools.experiment import Experiment


class OpenAIChatExperiment(Experiment):
    r"""
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.
    """

    PARAMETER_NAMES = (
        "model",
        "messages",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
    )

    def __init__(
        self,
        model: List[str],
        messages: List[List[Dict[str, str]]],
        use_scribe: bool = False,
        scribe_name: str = "Chat Experiment",
        temperature: Optional[List[float]] = [1.0],
        top_p: Optional[List[float]] = [1.0],
        n: Optional[List[int]] = [1],
        stream: Optional[List[bool]] = [False],
        stop: Optional[List[List[str]]] = [None],
        max_tokens: Optional[List[int]] = [float("inf")],
        presence_penalty: Optional[List[float]] = [0],
        frequency_penalty: Optional[List[float]] = [0],
        logit_bias: Optional[Dict] = [None],
    ):
        self.use_scribe = use_scribe
        if use_scribe:
            from hegel.scribe import HegelScribe

            self.completion_fn = HegelScribe(
                name=scribe_name, completion_fn=openai.ChatCompletion.create
            )
        else:
            self.completion_fn = openai.ChatCompletion.create
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_chat_completion_fn
        self.all_args = []
        self.all_args.append(model)
        self.all_args.append(messages)
        self.all_args.append(temperature)
        self.all_args.append(top_p)
        self.all_args.append(n)
        self.all_args.append(stream)
        self.all_args.append(stop)
        self.all_args.append(max_tokens)
        self.all_args.append(presence_penalty)
        self.all_args.append(frequency_penalty)
        self.all_args.append(logit_bias)
        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]]
