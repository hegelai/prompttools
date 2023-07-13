# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional
from huggingface_hub.inference_api import InferenceApi

import logging

from .experiment import Experiment

VALID_TASKS = ("text2text-generation", "text-generation", "summarization")

class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    """

    PARAMETER_NAMES = [
        "repo_id",
        "prompt",
        "task"
    ]

    def __init__(
        self,
        repo_id: List[str],
        prompt: List[str],
        task: List[str],
        use_scribe: bool = False,
        **kwargs: Dict[str,object]
    ):
        self.use_scribe = use_scribe
        self.completion_fn = self.hf_completion_fn

        self.all_args = []
        self.all_args.append(repo_id)
        self.all_args.append(prompt)
        self.all_args.append(task)
        for k, v in kwargs.items():
            self.PARAMETER_NAMES.append(k)
            self.all_args.append(v)
        super().__init__()

    def hf_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Hugging Face Hub helper function to make request
        """
        client = InferenceApi(repo_id=params['repo_id'],
                              token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
                              task=params['task'],
        )
        model_kwargs = {k: params[k] for k in params if k not in ["repo_id", "prompt", "task"]}
        response = client(inputs=params["prompt"], params=model_kwargs)
        logging.info(response)
        return response

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return output
