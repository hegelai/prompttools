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

EVALUATION_SYSTEM_PROMPT_BY_TYPE = {
    "RIGHT_OR_WRONG": """\
Determine whether or not the response is following directions.\
Your answer should either be "RIGHT" if the response follows directions,\
or "WRONG" if the model is not following directions.""",

    "SCORE": """\
Given the Fact and Answer, Evaluate the statement on a scale from 1 - 7.\
Please only respond with an integer from 1 - 7 with no other text.\
Lower score means the answer is factually wrong, higher score means\
the answer is correct. A medium score for uncertain but not wrong"""
}

EVALUATION_USER_TEMPLATE = """
PROMPT: {{prompt}}
RESPONSE: {{response}}
ANSWER:
"""

EVALUATE_AND_SCORE_USER_TEMPLATE = """
FACT: {{fact}}
ANSWER: {{answer}}
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

        self.evaluation_template = jinja2.Environment().from_string(
            EVALUATION_USER_TEMPLATE
        )

        self.evaluate_and_score_template = jinja2.Environment().from_string(
            EVALUATE_AND_SCORE_USER_TEMPLATE
        )

    @override
    def supports_model(self, model) -> bool:
        return model in self.supported_models

    @override
    def evaluate(self, model: str, prompt: str, response: str):
        if not os.environ["OPENAI_API_KEY"]:
            raise PromptToolsUtilityError

        eval_prompt = self.evaluation_template.render(prompt=prompt, response=response)
        response = openai.ChatCompletion.create(
            model=model, messages=self.get_messages("RIGHT/WRONG", eval_prompt)
        )

        return response["choices"][0]["message"]["content"]

    def evaluate_and_score(self, model: str, fact: str, answer: str):
        if not os.environ["OPENAI_API_KEY"]:
            raise PromptToolsUtilityError

        eval_prompt = self.evaluate_and_score_template.render(fact=fact, answer=answer)
        response = openai.ChatCompletion.create(
            model=model, messages=self.get_messages("SCORE", eval_prompt)
        )

        return response["choices"][0]["message"]["content"]

    def get_messages(self, evaluation_type: str, eval_prompt: str) -> list:
        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT_BY_TYPE[evaluation_type]},
            {"role": "user", "content": eval_prompt},
        ]

        return messages
