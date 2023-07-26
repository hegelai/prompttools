# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Dict

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None
    Anthropic = None


from prompttools.mock.mock import mock_anthropic_completion_fn
from .experiment import Experiment


class AnthropicCompletionExperiment(Experiment):
    r"""
    This class defines an experiment for Anthropic's completion API. It accepts lists for each argument
    passed into Anthropic's API, then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You should set ``os.environ["ANTHROPIC_API_KEY"] = YOUR_KEY`` in order to connect with Anthropic's API.

    Args:
        max_tokens_to_sample (list[int]):
            A list of integers representing The maximum number of tokens to generate before stopping.

        model (list[str]):
            the model(s) that will complete your prompt (e.g. "claude-2", "claude-instant-1")

        prompt (list[str]):
            Input prompt. For proper response generation you will need to format your prompt as follows:
            ``f"{HUMAN_PROMPT} USER_QUESTION {AI_PROMPT}"``, you can get built-in string by importing
            ``from anthropic HUMAN_PROMPT, AI_PROMPT``

        metadata (list):
            list of object(s) describing metadata about the request.

        stop_sequences (list[list[str]], optional):
            Sequences that will cause the model to stop generating completion text

        stream (list[bool], optional):
            Whether to incrementally stream the response using server-sent events.

        temperature (list[float], optional):
             The amount of randomness injected into the response

        top_k (list[int], optional):
            Only sample from the top K options for each subsequent token.

        top_p (list[float], optional):
            use nucleus sampling.

        timeout (list[float], optional):
           Override the client-level default timeout for this request, in seconds. Defaults to [600.0].
    """

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
        if anthropic is None or Anthropic is None:
            raise ModuleNotFoundError(
                "Package `anthropic` is required to be installed to use this experiment."
                "Please use `pip install anthropic` to install the package"
            )
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.completion_fn = self.anthropic_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_anthropic_completion_fn
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
            return self.client.completions.create(**input_args)
        except anthropic.APIConnectionError as e:
            logging.error("The server could not be reached")
            logging.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except anthropic.RateLimitError:
            logging.error("A 429 rate limit status code was received; we should back off a bit.")
        except anthropic.APIStatusError as e:
            logging.error("Another non-200-range status code was received")
            logging.error(e.status_code)
            logging.error(e.response)

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["message"]["content"] for choice in output["choices"]]

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]
