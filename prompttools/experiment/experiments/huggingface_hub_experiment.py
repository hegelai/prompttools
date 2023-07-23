# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List
from huggingface_hub.inference_api import InferenceApi

from prompttools.mock.mock import mock_hf_completion_fn

from .experiment import Experiment

VALID_TASKS = ("text2text-generation", "text-generation", "summarization")


class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    It accepts lists for each argument passed into Hugging Face Hub's API,
    then creates a cartesian product of those arguments, and gets results for each.
    """

    PARAMETER_NAMES = ["repo_id", "prompt", "task"]

    def __init__(
        self,
        repo_id: List[str],
        prompt: List[str],
        task: List[str] = ["text-generation"],
        **kwargs: Dict[str, object],
    ):
        self.completion_fn = self.hf_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_hf_completion_fn
        self.all_args = dict(repo_id=repo_id, prompt=prompt, task=task)
        for k, v in kwargs.items():
            self.PARAMETER_NAMES.append(k)
            self.all_args[k] = v
        super().__init__()

    def hf_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Hugging Face Hub helper function to make request
        """
        client = InferenceApi(
            repo_id=params["repo_id"],
            token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            task=params["task"],
        )
        model_kwargs = {k: params[k] for k in params if k not in ["repo_id", "prompt", "task"]}
        response = client(inputs=params["prompt"], params=model_kwargs)
        return response

    @staticmethod
    def _extract_responses(output: List[Dict[str, object]]) -> list[str]:
        return [choice["generated_text"] for choice in output][0]
