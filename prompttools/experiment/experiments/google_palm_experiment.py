# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

try:
    import google.generativeai as palm
except ImportError:
    palm = None

# from prompttools.mock.mock import mock_palm_completion_fn
from .experiment import Experiment
from typing import Optional, Union, Iterable
import os


class GooglePaLMCompletionExperiment(Experiment):
    r"""
    This class defines an experiment for Google PaLM's generate text API. It accepts lists for each argument
    passed into PaLM's API, then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You should set ``os.environ["GOOGLE_PALM_API_KEY"] = YOUR_KEY`` in order to connect with PaLM's API.

    Args:
        model (list[str]): Which model to call, as a string or a `types.Model`.

        prompt (list[str]): Free-form input text given to the model. Given a prompt, the model will
            generate text that completes the input text.

        temperature (list[float]): Controls the randomness of the output. Must be positive.
            Typical values are in the range: ``[0.0,1.0]``. Higher values produce a
            more random and varied response. A temperature of zero will be deterministic.

        candidate_count (list[int]): The **maximum** number of generated response messages to return.
            This value must be between ``[1, 8]``, inclusive. If unset, this will default to ``1``.

        max_output_tokens (list[int]): Maximum number of tokens to include in a candidate. Must be greater
            than zero. If unset, will default to ``64``.

        top_k (list[float]): The API uses combined nucleus and top-k sampling.
            ``top_k`` sets the maximum number of tokens to sample from on each step.

        top_p (list[float]): The API uses combined nucleus and top-k sampling. ``top_p`` configures the nucleus
            sampling. It sets the maximum cumulative probability of tokens to sample from.

        safety_settings (list[Iterable[palm.types.SafetySettingDict]]): A list of unique ``types.SafetySetting``
            instances for blocking unsafe content.

        stop_sequences (list[Union[str, Iterable[str]]]): A set of up to 5 character sequences that will stop output
            generation. If specified, the API will stop at the first appearance of a stop sequence.
    """

    def __init__(
        self,
        model: list[str],
        prompt: list[str],
        temperature: list[Optional[float]] = [None],
        candidate_count: list[Optional[int]] = [None],
        max_output_tokens: list[Optional[int]] = [None],
        top_p: list[Optional[float]] = [None],
        top_k: list[Optional[float]] = [None],
        safety_settings: list[Optional[Iterable["palm.types.SafetySettingDict"]]] = [None],
        stop_sequences: list[Union[str, Iterable[str]]] = [None],
    ):
        if palm is None:
            raise ModuleNotFoundError(
                "Package `google.generativeai` is required to be installed to use PaLM API in this experiment."
                "Please use `pip install google.generativeai` to install the package"
            )
        palm.configure(api_key=os.environ["GOOGLE_PALM_API_KEY"])
        self.completion_fn = self.palm_completion_fn
        # if os.getenv("DEBUG", default=False):
        #     self.completion_fn = mock_palm_completion_fn()
        self.all_args = dict(
            model=model,
            prompt=prompt,
            temperature=temperature,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            safety_settings=safety_settings,
            stop_sequences=stop_sequences,
        )
        super().__init__()

    def palm_completion_fn(self, **input_args):
        return palm.generate_text(**input_args)

    @staticmethod
    def _extract_responses(completion_response: "palm.text.text_types.Completion") -> list[str]:
        # `# completion_response.result` will return the top response
        return [candidate["output"] for candidate in completion_response.candidates]
