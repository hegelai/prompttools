# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional

import logging

from langchain import HuggingFaceHub, LLMChain

from .experiment import Experiment
from prompttools.mock.mock import mock_chat_completion_fn


class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    """

    PARAMETER_NAMES = (
        "repo_id",
        "max_length",
        "min_length",
        "max_time",
        "temperature",
        "top_k",
        "top_p",
    )

    def __init__(
        self,
        repo_id: List[str] = ["gpt2"],
        max_length: List[int] = [20],
        min_length: List[int] = [0],
        max_time: List[Optional[float]] = [None],
        temperature: List[float] = [1.0],
        top_k: List[int] = [50],
        top_p: List[float] = [0.999],
        use_scribe: bool = False,
    ):
        self.hf_fn = HuggingFaceHub
        self.use_scribe = use_scribe
        # Make this optional?
        self.completion_fn = mock_chat_completion_fn

        self.all_args = []
        self.all_args.append(repo_id)
        self.all_args.append(max_length)
        self.all_args.append(min_length)
        self.all_args.append(max_time)
        self.all_args.append(temperature)
        self.all_args.append(top_k)
        self.all_args.append(top_p)
        super().__init__()

        self.hf = True

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]]
