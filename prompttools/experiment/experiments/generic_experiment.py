# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from typing import Callable

from .experiment import Experiment


def _response_field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


class _CallableExperiment(Experiment):
    def __init__(self, completion_fn: Callable, **parameters: list):
        if not callable(completion_fn):
            raise TypeError("completion_fn must be callable")
        for name, values in parameters.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name} must be a nonempty list of parameter choices")
        self.completion_fn = completion_fn
        self.all_args = parameters
        super().__init__()

    @classmethod
    def initialize(cls, completion_fn: Callable, test_parameters: dict[str, list], frozen_parameters: dict):
        """Create a callable experiment with tested and frozen API parameters.

        The callable is supplied separately so it is never wrapped in a list or
        passed back to the API as a request argument.
        """
        frozen_parameters = {name: [value] for name, value in frozen_parameters.items()}
        return cls(completion_fn, **test_parameters, **frozen_parameters)

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]


class CompletionExperiment(_CallableExperiment):
    """Run an OpenAI-compatible completion callable against a parameter grid.

    ``completion_fn`` is a synchronous function accepting keyword arguments,
    such as a configured client's ``completions.create`` method. The caller
    owns authentication, endpoint selection, and client configuration.
    Responses may be mappings or SDK objects with ``choices[0].text``.

    ``model``, ``prompt``, and each additional API parameter must be nonempty
    lists of choices. No provider-specific defaults are added. For example::

        experiment = CompletionExperiment(
            client.completions.create,
            model=["local-model"],
            prompt=["Write a greeting", "Write a farewell"],
            temperature=[0.0, 0.5],
        )
        experiment.run()
    """

    def __init__(self, completion_fn: Callable, model: list[str], prompt: list[str], **parameters: list):
        self.prompt_keys = prompt
        super().__init__(completion_fn, model=model, prompt=prompt, **parameters)

    @staticmethod
    def _extract_responses(output) -> str:
        return _response_field(_response_field(output, "choices")[0], "text")


class ChatCompletionExperiment(_CallableExperiment):
    """Run an OpenAI-compatible chat callable against a parameter grid.

    Each item of ``messages`` is a complete conversation, for example
    ``[[{"role": "user", "content": "Hello"}]]``. Other API parameters follow
    :class:`CompletionExperiment`'s list-of-choices convention. Responses may
    be mappings or SDK objects with ``choices[0].message.content``.

    The callable and its client remain under the caller's control::

        experiment = ChatCompletionExperiment(
            client.chat.completions.create,
            model=["local-model"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[0.0, 0.5],
        )
        experiment.run()

    These experiments consume complete responses; streaming callables need a
    caller-provided adapter that collects the stream into a complete response.
    """

    def __init__(self, completion_fn: Callable, model: list[str], messages: list[list[dict]], **parameters: list):
        self.prompt_keys = messages
        super().__init__(completion_fn, model=model, messages=messages, **parameters)

    @staticmethod
    def _is_chat():
        return True

    @staticmethod
    def _extract_responses(output) -> str:
        choice = _response_field(output, "choices")[0]
        return _response_field(_response_field(choice, "message"), "content")
