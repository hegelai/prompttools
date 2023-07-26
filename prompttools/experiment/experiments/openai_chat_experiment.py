# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional
import openai

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.mock.mock import mock_openai_chat_completion_fn
from .experiment import Experiment


class OpenAIChatExperiment(Experiment):
    r"""
    This class defines an experiment for OpenAI's chat completion API.
    It accepts lists for each argument passed into OpenAI's API, then creates
    a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
    """

    def __init__(
        self,
        model: List[str],
        messages: List[List[Dict[str, str]]] | List[PromptSelector],
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
        self.completion_fn = openai.ChatCompletion.create
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_openai_chat_completion_fn

        # If we are using a prompt selector, we need to render
        # messages, as well as create prompt_keys to map the messages
        # to corresponding prompts in other models.
        if isinstance(messages[0], PromptSelector):
            self.prompt_keys = {
                str(selector.for_openai_chat()[-1]["content"]): selector.for_llama() for selector in messages
            }
            messages = [selector.for_openai_chat() for selector in messages]
        else:
            self.prompt_keys = messages

        self.all_args = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_token=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
        )
        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]][0]

    def _is_chat(self):
        return True

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        return [self.prompt_keys[str(combo["messages"][-1]["content"])] for combo in self.argument_combos]
