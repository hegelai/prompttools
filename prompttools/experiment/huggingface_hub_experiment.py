# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional

import logging

from prompttools.mock.mock import mock_chat_completion_fn
from .experiment import Experiment


class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    """

    PARAMETER_NAMES = (
        "repo_id",
        "temperature",
        "max_length",
        "template",
        "question",
        "expected",
    )

    def __init__(
        self,
        repo_id: List[str] = [],
        temperature: List[float] = [],
        max_length: List[int] = [],
        template: List[str] = ["""Question: {question}
        Answer: """],
        question: List[str] = [""],
        expected: List[str] = [""],
        use_scribe: bool = False,
    ):
        # Placeholder. Should this be optional in experiment class?
        self.completion_fn = mock_chat_completion_fn

        self.use_scribe = use_scribe

        # Notify Experiments this is HF run
        self.hf = True

        self.all_args = []
        self.all_args.append(repo_id)
        self.all_args.append(temperature)
        self.all_args.append(max_length)
        self.all_args.append(template)
        self.all_args.append(question)
        self.all_args.append(expected)
        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]]
