# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional

import logging

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

from .experiment import Experiment
from prompttools.mock.mock import mock_chat_completion_fn


class HuggingFaceHubExperiment(Experiment):
    r"""
    Experiment for Hugging Face Hub's API.
    """

    PARAMETER_NAMES = (
        "repo_id",
        "messages",
        "max_length",
        "min_length",
        "max_time",
        "temperature",
        "top_k",
        "top_p",
    )

    def __init__(
        self,
        messages: List[List[Dict[str, str]]],
        repo_id: List[str] = ["gpt2"],
        max_length: List[int] = [20],
        min_length: List[int] = [0],
        max_time: List[Optional[float]] = [None],
        temperature: List[float] = [1.0],
        top_k: List[int] = [50],
        top_p: List[float] = [0.999],
        template: str = """Question: {question}
        Answer: """,
        input_variables: List[str] = ["question"],
        question: str = "Who was the first president?",
        context: str = "The first president",
        expected: str = "George Washington",

        use_scribe: bool = False,
    ):
        self.use_scribe = use_scribe
        self.completion_fn = self.hf_completion_fn

        self.all_args = []
        self.all_args.append(repo_id)
        self.all_args.append(messages)
        self.all_args.append(max_length)
        self.all_args.append(min_length)
        self.all_args.append(max_time)
        self.all_args.append(temperature)
        self.all_args.append(top_k)
        self.all_args.append(top_p)
        super().__init__()

        self.hf = True
        self.prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.expected = expected
        self.query = question if input_variables == ["question"] else context
        self.query_type = "question" if input_variables == ["question"] else "context"

    def hf_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        print("params: ", params)
        model_kwargs = {k: params[k] for k in params if k != "repo_id"}
        llm = HuggingFaceHub(repo_id=params["repo_id"], model_kwargs=model_kwargs)
        llm_chain = LLMChain(prompt=self.prompt, llm=llm)
        return llm_chain.run(self.query)

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return output
