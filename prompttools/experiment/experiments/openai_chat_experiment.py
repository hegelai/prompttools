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
        - For detailed description of the input arguments, please reference at OpenAI's chat completion API.

    Args:
        model (list[str]): list of ID(s) of the model(s) to use
        messages (list[dict]): A list of messages comprising the conversation so far. Each message is represented as a
            dictionary with the following keys: ``role: str``, ``content: str``.

        temperature (list[float]):
            Defaults to [1.0]. What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
            the output more random, while lower values like 0.2 will make it more focused and deterministic.

        top_p (list[float]):
            Defaults to [1.0]. An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.

        n (list[int]):
            Defaults to [1]. How many chat completion choices to generate for each input message.

        stream (list[bool]):
            Defaults to [False]. If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent
            as data-only server-sent events as they become available, with the stream terminated by a data: [DONE]
            message.

        stop (list[list[str]]):
            Defaults to [None]. Up to 4 sequences where the API will stop generating further tokens.

        max_tokens (list[int]):
            Defaults to [inf]. The maximum number of tokens to generate in the chat completion.

        presence_penalty (list[float]):
            Defaults to [0.0]. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
            they appear in the text so far, increasing the model's likelihood to talk about new topics.

        frequency_penalty (list[float]):
            Defaults to [0.0]. Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to repeat the same line
            verbatim.

        logit_bias (list[dict]):
            Defaults to [None]. Modify the likelihood of specified tokens appearing in the completion. Accepts a
            json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value
            from -100 to 100.
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
        presence_penalty: Optional[List[float]] = [0.0],
        frequency_penalty: Optional[List[float]] = [0.0],
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
        return [choice["message"]["content"].replace("\n", "<br />") for choice in output["choices"]][0]

    def _is_chat(self):
        return True

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        return [self.prompt_keys[str(combo["messages"][-1]["content"])] for combo in self.argument_combos]
