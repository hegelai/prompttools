# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Optional


from .experiment import Experiment


try:
    import mistralai
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    mistralai = None
    MistralClient = None
    ChatMessage = None


class MistralChatCompletionExperiment(Experiment):
    r"""
    This class defines an experiment for Mistral's  chatcompletion API. It accepts lists for each argument
    passed into the API, then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You should set ``os.environ["MISTRAL_API_KEY"] = YOUR_KEY`` in order to connect with Mistral's API.

    Args:
        model (list[str]):
            the model(s) that will complete your prompt (e.g. "mistral-tiny")

        messages (list[ChatMessage]):
            Input prompts (using Mistral's Python library). The first prompt role should be `user` or `system`.

        temperature (list[float], optional):
             The amount of randomness injected into the response

        top_p (list[float], optional):
            use nucleus sampling.

        max_tokens (list[int]):
            The maximum number of tokens to generate in the completion.

        safe_prompt (list[bool]):
            Whether to inject a safety prompt before all conversations.

        random_seed (list[int], optional):
           The seed to use for random sampling. If set, different calls will generate deterministic results.
    """

    url = "https://api.mistral.ai/v1/chat/completions"

    def __init__(
        self,
        model: list[str],
        messages: list[str],
        temperature: list[float] = [None],
        top_p: list[float] = [None],
        max_tokens: list[Optional[int]] = [None],
        safe_prompt: list[bool] = [False],
        random_seed: list[Optional[int]] = [None],
    ):
        if mistralai is None:
            raise ModuleNotFoundError(
                "Package `mistralai` is required to be installed to use this experiment."
                "Please use `pip install mistralai` to install the package"
            )
        self.client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
        self.completion_fn = self.mistral_completion_fn

        self.all_args = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            safe_prompt=safe_prompt,
            random_seed=random_seed,
        )
        super().__init__()

    def mistral_completion_fn(self, **input_args):
        response = self.client.chat(**input_args)
        return response

    @staticmethod
    def _extract_responses(response) -> list[str]:
        return response.choices[0].message.content

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        return [combo["messages"] for combo in self.argument_combos]
