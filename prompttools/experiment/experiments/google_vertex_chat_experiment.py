# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

try:
    from vertexai.preview.language_models import ChatModel, InputOutputTextPair
except ImportError:
    ChatModel = None
    InputOutputTextPair = None

from .experiment import Experiment
from typing import Optional
import copy


class GoogleVertexChatCompletionExperiment(Experiment):
    r"""
    This class defines an experiment for Google Vertex AI's chat API. It accepts lists for each argument
    passed into Vertex AI's API, then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You need to set up your Google Vertex AI credentials properly before executing this experiment. One option
          is to execute on Google Cloud's Colab.

    Args:
        model (list[str]): Which model to call, as a string or a ``types.Model`` (e.g. ``'models/text-bison-001'``).

        message (list[str]): Message for the chat model to respond.

        context (list[str]): Context shapes how the model responds throughout the conversation. For example,
            you can use context to specify words the model can or cannot use,
            topics to focus on or avoid, or the response format or style.

        examples (list[list['InputOutputTextPair']]): Examples for the model to learn how to
            respond to the conversation.

        temperature (list[float]): Controls the randomness of the output. Must be positive.
            Typical values are in the range: ``[0.0, 1.0]``. Higher values produce a
            more random and varied response. A temperature of zero will be deterministic.

        max_output_tokens (list[int]): Maximum number of tokens to include in a candidate. Must be greater
            than zero. If unset, will default to ``64``.

        top_k (list[float]): The API uses combined nucleus and top-k sampling.
            ``top_k`` sets the maximum number of tokens to sample from on each step.

        top_p (list[float]): The API uses combined nucleus and top-k sampling. ``top_p`` configures the nucleus
            sampling. It sets the maximum cumulative probability of tokens to sample from.

        stop_sequences (list[Union[str, Iterable[str]]]): A set of up to 5 character sequences that will stop output
            generation. If specified, the API will stop at the first appearance of a stop sequence.
    """

    def __init__(
        self,
        model: list[str],
        message: list[str],
        context: list[Optional[str]] = [None],
        examples: list[Optional[list[InputOutputTextPair]]] = [None],
        temperature: list[Optional[float]] = [None],
        max_output_tokens: list[Optional[int]] = [None],
        top_p: list[Optional[float]] = [None],
        top_k: list[Optional[int]] = [None],
        stop_sequences: list[list[str]] = [None],
    ):
        if ChatModel is None:
            raise ModuleNotFoundError(
                "Package `vertexai` is required to be installed to use Google Vertex API in this experiment."
                "Please use `pip install google-cloud-aiplatform` to install the package"
            )

        self.completion_fn = self.vertex_chat_completion_fn

        self.all_args = dict(
            model=model,
            message=message,
            context=context,
            examples=examples,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )
        super().__init__()

    def vertex_chat_completion_fn(self, **input_args):
        chat_model = ChatModel.from_pretrained(model_name=input_args["model"])
        message = input_args["message"]
        params = copy.deepcopy(input_args)
        del params["model"], params["message"]
        chat = chat_model.start_chat(**params)
        return chat.send_message(message)

    @staticmethod
    def _extract_responses(response) -> list[str]:
        # `response.text` will return the top response
        return response.text

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        return [combo["message"] for combo in self.argument_combos]
