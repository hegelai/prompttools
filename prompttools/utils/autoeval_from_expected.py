# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import jinja2
import pandas
from .model_evaluators.EvaluatorUtils import get_evaluator_for_model

EVALUATION_SYSTEM_PROMPT = """
You are a grader evaluating responses to math questions.
Given the PROMPT and EXPECTED, evaluate the ACTUAL answer.
The ACTUAL answer should be the same as the EXPECTED.
You should grade the response as either RIGHT or WRONG.
If the ACTUAL answer is the same as the EXPECTED, mark it RIGHT.
Otherwise, mark it WRONG.
"""

EVALUATION_USER_TEMPLATE = """
PROMPT: {{prompt}}
EXPECTED: {{expected}}
ACTUAL: {{actual}}
ANSWER:
"""


def _get_user_prompt(prompt: str, expected: str, response: str):
    environment = jinja2.Environment()
    template = environment.from_string(EVALUATION_USER_TEMPLATE)
    user_prompt = template.render({"prompt": prompt, "expected": expected, "actual": response})
    return user_prompt


def compute(prompt: str, expected: str, response: str, model: str = "gpt-4") -> float:
    r"""
    Uses a high quality chat model, like GPT-4, to automatically evaluate a given
    prompt/response pair. Outputs can be 0 or 1.

    Args:
        prompt (str): The input prompt.
        response (str): The model response.
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    evaluation = get_evaluator_for_model(model).evaluate(
        model, EVALUATION_SYSTEM_PROMPT, _get_user_prompt(prompt, expected, response)
    )
    return 1.0 if "RIGHT" in evaluation["choices"][0]["message"]["content"] else 0.0


def evaluate(prompt: str, response: str, metadata: dict, expected: str) -> float:
    r"""
    Uses auto-evaluation to score the model response.
    """
    return compute(prompt, expected, response)


def autoeval_from_expected_response(
    row: pandas.core.series.Series, expected: str, prompt_column_name: str, response_column_name: str = "response"
):
    prompt = row[prompt_column_name]
    response = row[response_column_name]
    return compute(prompt, expected, response)
