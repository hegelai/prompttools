# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Dict, List, Optional, Union

import openai

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.mock.mock import mock_minimax_chat_completion_fn
from .experiment import Experiment


MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MiniMaxChatExperiment(Experiment):
    r"""
    This class defines an experiment for MiniMax's chat completion API.
    It accepts lists for each argument passed into the API, then creates
    a cartesian product of those arguments, and gets results for each.

    MiniMax provides an OpenAI-compatible chat completions endpoint, so this
    experiment uses the ``openai`` library with a custom ``base_url``.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You should set ``os.environ["MINIMAX_API_KEY"] = YOUR_KEY`` in order to connect
          with MiniMax's API.

    Args:
        model (list[str]):
            The model(s) to use. Available models include ``"MiniMax-M2.7"``,
            ``"MiniMax-M2.7-highspeed"``, ``"MiniMax-M2.5"``, and
            ``"MiniMax-M2.5-highspeed"``.

        messages (list[list[dict[str, str]]]):
            Input prompts using the OpenAI message format. Each message is a
            dictionary with ``role`` and ``content`` keys.

        temperature (list[float], optional):
            The sampling temperature. MiniMax requires values in (0.0, 1.0].
            Defaults to ``[1.0]``.

        top_p (list[float], optional):
            Nucleus sampling parameter. Defaults to ``[1.0]``.

        max_tokens (list[int], optional):
            The maximum number of tokens to generate. Defaults to ``[None]``.

        stop (list[list[str]], optional):
            Up to 4 sequences where the API will stop generating further tokens.
            Defaults to ``[None]``.
    """

    url = "https://api.minimax.io/v1/chat/completions"

    def __init__(
        self,
        model: List[str] = ["MiniMax-M2.5"],
        messages: Union[List[List[Dict[str, str]]], List[PromptSelector]] = [],
        temperature: Optional[List[float]] = [1.0],
        top_p: Optional[List[float]] = [1.0],
        max_tokens: Optional[List[Optional[int]]] = [None],
        stop: Optional[List[Optional[List[str]]]] = [None],
    ):
        self.client = openai.OpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url=MINIMAX_API_BASE,
        )
        self.completion_fn = self.minimax_completion_fn

        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_minimax_chat_completion_fn

        # Handle PromptSelector
        if len(messages) > 0 and isinstance(messages[0], PromptSelector):
            self.prompt_keys = {
                str(selector.for_openai_chat()[-1]["content"]): selector.for_llama() for selector in messages
            }
            messages = [selector.for_openai_chat() for selector in messages]
        else:
            self.prompt_keys = messages

        # Clamp temperature: MiniMax requires (0.0, 1.0]
        clamped_temperature = []
        for t in temperature:
            if t is not None:
                t = max(0.01, min(t, 1.0))
            clamped_temperature.append(t)

        self.all_args = dict(
            model=model,
            messages=messages,
            temperature=clamped_temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        super().__init__()

    def minimax_completion_fn(self, **input_args):
        response = self.client.chat.completions.create(**input_args)
        return response

    @staticmethod
    def _extract_responses(response) -> str:
        return response.choices[0].message.content

    @staticmethod
    def _is_chat():
        return True

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        if isinstance(self.prompt_keys, dict):
            return [self.prompt_keys[str(combo["messages"][-1]["content"])] for combo in self.argument_combos]
        return [combo["messages"] for combo in self.argument_combos]
