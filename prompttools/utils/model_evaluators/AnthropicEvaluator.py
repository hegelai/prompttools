# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from overrides import override
from .ModelEvaluator import ModelEvaluator
import jinja2
import anthropic
import os

ANTHROPIC_API_AUTOEVAL_TEMPLATE = """
{{HUMAN_PROMPT}} {{EVALUATION_SYSTEM_PROMPT}}
{{USER_MESSAGE}} {{AI_PROMPT}}
"""


class AnthropicEvaluator(ModelEvaluator):
    def __init__(self) -> None:
        self.client = None
        self.supported_models = ["claude-1", "claude-2"]

    def supports_model(self, model: str):
        return model in self.supports_model(model)

    @override
    def evaluate(self, model: str, evaluation_system_prompt: str, user_message: str):
        if anthropic is None:
            raise ModuleNotFoundError(
                "Package `anthropic` is required to be installed to use this experiment."
                "   Please use `pip install anthropic` to install the package"
            )

        if not os.environ["ANTHROPIC_API_KEY"]:
            raise RuntimeError("Missing API key for evaluation.")

        if not self.client:
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        environment = jinja2.Environment()
        template = environment.from_string(ANTHROPIC_API_AUTOEVAL_TEMPLATE)
        eval_prompt = template.render(
            {
                "HUMAN_PROMPT": anthropic.HUMAN_PROMPT,
                "EVALUATION_SYSTEM_PROMPT": evaluation_system_prompt,
                "USER_MESSAGE": user_message,
                "AI_PROMPT": anthropic.AI_PROMPT,
            }
        )

        response = self.client.completions.create(max_tokens_to_sample=100, model=model, prompt=eval_prompt)

        return response.completion
