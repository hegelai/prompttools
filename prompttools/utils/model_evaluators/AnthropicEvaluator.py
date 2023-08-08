# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from functools import cached_property
from overrides import override
from .ModelEvaluator import ModelEvaluator
import jinja2
import anthropic
import os

EVALUATE_AND_SCORE_PROMPT_TEMPLATE = """\
{{HUMAN_PROMPT}} Given the fact {{fact}}
Evaluate the following Answer on a scale from 1 - 7.\
Please only respond with an integer from 1 - 7 with no other text.
Lower score means the answer is factually wrong, higher score means the answer\ 
is correct. A medium score for uncertain but not wrong.

Answer: {{answer}}
{{AI_PROMPT}}
"""

EVALUATE_RIGHT_OR_WRONG_PROMPT_TEMPLATE = """\
{{HUMAN_PROMPT}} Determine whether or not the response is following directions.\
Your answer should either be "RIGHT" if the response follows directions,\
or "WRONG" if the model is not following directions.

PROMPT: {{prompt}}
RESPONSE: {{response}}
{{AI_PROMPT}}
"""


class AnthropicEvaluator(ModelEvaluator):
    def __init__(self) -> None:
        self.client = None
        self.supported_models = ["claude-1", "claude-2"]
        self.right_or_wrong_evaluation_template = jinja2.Environment().from_string(
            EVALUATE_RIGHT_OR_WRONG_PROMPT_TEMPLATE
        )

        self.evaluate_and_score_template = jinja2.Environment().from_string(
            EVALUATE_AND_SCORE_PROMPT_TEMPLATE
        )

    @cached_property
    def get_client(self):
        try:
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except:
            self.client = None

    def supports_model(self, model: str):
        return model in self.supports_model(model)

    @override
    def evaluate(self, model: str, prompt: str, response: str):
        client = self.validate_and_get_client()
        eval_prompt = self.right_or_wrong_evaluation_template.render(
            HUMAN_PROMPT=anthropic.HUMAN_PROMPT,
            prompt=prompt,
            response=response,
            AI_PROMPT=anthropic.AI_PROMPT
        )

        response = client.completions.create(max_tokens_to_sample=100, model=model, prompt=eval_prompt)
        return response.completion

    @override
    def evaluate_and_score(self, model: str, fact: str, answer: str):
        client = self.validate_and_get_client()
        eval_prompt = self.evaluate_and_score_template.render(
            HUMAN_PROMPT=anthropic.HUMAN_PROMPT,
            fact=fact,
            answer=answer,
            AI_PROMPT=anthropic.AI_PROMPT
        )

        response = client.completions.create(max_tokens_to_sample=100, model=model, prompt=eval_prompt)
        return response.completion

    def validate_and_get_client(self) -> anthropic.Anthropic:
        if anthropic is None:
            raise ModuleNotFoundError(
                "Package `anthropic` is required to be installed to use this experiment."
                "Please use `pip install anthropic` to install the package"
            )

        if not os.environ["ANTHROPIC_API_KEY"]:
            raise RuntimeError("Missing API key for evaluation.")

        client = self.validate_and_get_client()
        if client == None:
            raise RuntimeError("Could not connect to Anthropic Client")

        return self.get_client()
