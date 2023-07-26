# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict
import anthropic
from anthropic import Anthropic

# from prompttools.mock.mock import mock_anthropic_completion_fn
from .experiment import Experiment


r"""
    This class defines an experiment for Anthropic's completion API.
    It accepts lists for each argument passed into Anthropic's API, then creates
    a cartesian product of those arguments, and gets results for each.

    Note that all arguments here should be a ``list``, because the experiment
    will try all possible combination of the input arguments.
    """


class AnthropicCompletionExperiment(Experiment):
    def __init__(
        self,
        max_tokens_to_sample: list[int],
        model: list[str],
        prompt: list[str],
        metadata: list = [],
        stop_sequences: list[list[str]] = [],
        stream: list[bool] = [False],
        temperature: list[float] = [],
        top_k: list[int] = [],
        top_p: list[float] = [],
        timeout: list[float] = [600.0],
        # These are extra parameters not available via kwargs
        # extra_headers=None,
        # extra_query=None,
        # extra_body=None,
    ):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.completion_fn = self.client.completions.create
        # if os.getenv("DEBUG", default=False):
        #     self.completion_fn = mock_anthropic_completion_fn
        self.all_args = dict(
            max_tokens_to_sample=max_tokens_to_sample,
            model=model,
            prompt=prompt,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            timeout=timeout,
        )
        super().__init__()

    def anthropic_completion_fn(self, **input_args):
        try:
            self.client.completions.create(**input_args)
        except anthropic.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except anthropic.RateLimitError:
            print("A 429 rate limit status code was received; we should back off a bit.")
        except anthropic.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]][0]

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]
