# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

try:
    import google.generativeai as genai
    from google.generativeai.types import content_types
    from google.generativeai.types import generation_types
    from google.generativeai.types import safety_types
except ImportError:
    genai = None
    content_types, generation_types, safety_types = None, None, None


from .experiment import Experiment
from typing import Optional
import copy


class GoogleGeminiChatCompletionExperiment(Experiment):
    r"""
    This class defines an experiment for Google GenAI's chat API. It accepts lists for each argument
    passed into Vertex AI's API, then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - You need to set up your Google Vertex AI credentials properly before executing this experiment. One option
          is to execute on Google Cloud's Colab.

    Args:
        model (list[str]): Which model to call, as a string or a ``types.Model`` (e.g. ``'models/text-bison-001'``).

        contents (list[content_types]): Message for the chat model to respond.

        generation_config (list[generation_types]): Configurations for the generation of the model.

        safety_settings (list[safety_types]): Configurations for the safety features of the model.
    """

    def __init__(
        self,
        model: list[str],
        contents: list["content_types.ContentsType"],
        generation_config: list[Optional["generation_types.GenerationConfigType"]] = [None],
        safety_settings: list[Optional["safety_types.SafetySettingOptions"]] = [None],
    ):
        if genai is None:
            raise ModuleNotFoundError(
                "Package `google-generativeai` is required to be installed to use Google GenAI API in this experiment."
                "Please use `pip install google-generativeai` to install the package or run this in Google Colab."
            )

        self.completion_fn = self.google_text_completion_fn

        self.all_args = dict(
            model=model,
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        super().__init__()

    def google_text_completion_fn(self, **input_args):
        params = copy.deepcopy(input_args)
        model = genai.GenerativeModel(input_args["model"])
        del params["model"]
        response = model.generate_content(**params)
        return response

    @staticmethod
    def _extract_responses(response) -> list[str]:
        # `response.text` will return the top response
        return response.text

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    def _get_prompts(self):
        return [combo["message"] for combo in self.argument_combos]
