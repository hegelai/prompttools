# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import jinja2
from overrides import override

from prompttools.utils.error import PromptToolsUtilityError
from .ModelEvaluator import ModelEvaluator
import openai

OPENAI_EVAL_PROMPT = """
{{USER_MESSAGE}}
ANSWER:
"""


class GptEvaluator(ModelEvaluator):
    def __init__(self) -> None:
        # source: https://platform.openai.com/docs/models/model-endpoint-compatibility
        self.supported_models = [
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
        ]

    @override
    def supports_model(self, model) -> bool:
        return model in self.supported_models

    @override
    def evaluate(self, model: str, evaluation_system_prompt: str, user_message: str):
        if not os.environ["OPENAI_API_KEY"]:
            raise PromptToolsUtilityError

        response = openai.ChatCompletion.create(
            model=model, messages=self.get_messages(evaluation_system_prompt, user_message)
        )
        return response["choices"][0]["message"]["content"]

    def get_messages(self, evaluation_system_prompt, user_message) -> list:
        environment = jinja2.Environment()
        template = environment.from_string(OPENAI_EVAL_PROMPT)
        eval_prompt = template.render(
            {
                "USER_MESSAGE": user_message,
            }
        )

        messages = [
            {"role": "system", "content": evaluation_system_prompt},
            {"role": "user", "content": eval_prompt},
        ]

        return messages
