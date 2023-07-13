# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional

import logging

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .experiment import Experiment


class LocalModelExperiment(Experiment):
    r"""
    Experiment for local models.
    """
    
    PARAMETER_NAMES = (
        "model_path",
        "messages",
        "suffix",
        "max_tokens",
        "temperature",
        "top_p",
        # "logprobs", removed until logits_all support included
        "repeat_penalty",
        "top_k",
    )

    def __init__(
        self,
        model_path: List[str],
        messages: List[List[Dict[str, str]]],
        suffix: List[Optional[str]] = [" "],
        max_tokens: List[Optional[int]] = [256],
        temperature: List[Optional[float]] = [0.8],
        top_p: List[Optional[float]] = [0.95],
        echo: Optional[bool] = False,
        stop: Optional[List[str]] = [],
        repeat_penalty: List[Optional[float]] = [1.1],
        top_k: List[Optional[int]] = [40],
        verbose: bool = False,
        template = """Question: {question}

        Answer: """,
        question: str = "Who was the first president?",
        context: str = "The first president",
        input_variables: List[str] = ["question"],
        use_scribe: bool = False,
        local_model_type: str = "llama",
    ):
        self.use_scribe = use_scribe
        if local_model_type == "llama":
            self.completion_fn = self.llama_completion_fn
        self.echo = echo
        self.stop = stop
        self.verbose = verbose

        self.all_args = []
        self.all_args.append(model_path)
        self.all_args.append(messages)
        self.all_args.append(suffix)
        self.all_args.append(max_tokens)
        self.all_args.append(temperature)
        self.all_args.append(top_p)
        self.all_args.append(repeat_penalty)
        self.all_args.append(top_k)

        print(self.all_args)
        
        super().__init__()

        self.prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.query = question if input_variables == ["question"] else context
        self.query_type = "question" if input_variables == ["question"] else "context"
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    def llama_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        llm = LlamaCpp(
                    model_path=params["model_path"],
                    suffix=params["suffix"],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    echo=self.echo,
                    stop=self.stop,
                    repeat_penalty=params["repeat_penalty"],
                    top_k=params["top_k"],
                    verbose=self.verbose,
                    callback_manager=self.callback_manager,
                )
        
        llm_chain = LLMChain(prompt=self.prompt, llm=llm)
        return llm_chain.run(self.query)
    
    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return output