# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional
import openai
import logging

from prompttools.mock.mock import mock_completion_fn
from prompttools.experiment.experiment import Experiment
from dialectic.api.client_api.dialectic_scribe import (
    DialecticScribe,
)  # TODO: Switch to `hegel`


class OpenAICompletionExperiment(Experiment):
    r"""
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.
    """

    PARAMETER_NAMES = (
        "model",
        "prompt",
        "suffix",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "logprobs",
        "echo",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "best_of",
        "logit_bias",
    )

    def __init__(
        self,
        model: List[str],
        prompt: List[str],
        use_scribe: bool = False,
        scribe_name: str = "Completion Experiment",
        suffix: Optional[List[str]] = [None],
        max_tokens: Optional[List[int]] = [float("inf")],
        temperature: Optional[List[float]] = [1.0],
        top_p: Optional[List[float]] = [1.0],
        n: Optional[List[int]] = [1],
        stream: Optional[List[bool]] = [False],
        logprobs: Optional[List[int]] = [None],
        echo: Optional[List[bool]] = [False],
        stop: Optional[List[List[str]]] = [None],
        presence_penalty: Optional[List[float]] = [0],
        frequency_penalty: Optional[List[float]] = [0],
        best_of: Optional[List[int]] = [1],
        logit_bias: Optional[Dict] = [None],
    ):
        self.use_scribe = use_scribe
        if use_scribe:
            self.completion_fn = DialecticScribe(
                name=scribe_name, completion_fn=openai.Completion.create
            )
        else:
            self.completion_fn = openai.Completion.create

        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_completion_fn
        self.all_args = []
        self.all_args.append(model)
        self.all_args.append(prompt)
        self.all_args.append(suffix)
        self.all_args.append(max_tokens)
        self.all_args.append(temperature)
        self.all_args.append(top_p)
        self.all_args.append(n)
        self.all_args.append(stream)
        self.all_args.append(logprobs)
        self.all_args.append(echo)
        self.all_args.append(stop)
        self.all_args.append(presence_penalty)
        self.all_args.append(frequency_penalty)
        self.all_args.append(best_of)
        self.all_args.append(logit_bias)
        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["text"] for choice in output["choices"]]
