# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict, List
import openai
import jinja2
from .error import PromptToolsUtilityError

EVALUATION_SYSTEM_PROMPT = """
You are a grader evaluating responses to math questions.
Given the PROMPT and EXPECTED, evalaute the ACTUAL answer.
You should grade the response as either RIGHT or WRONG.
"""

EVALUATION_USER_TEMPLATE = """
PROMPT: {{prompt}}
EXPECTED: {{expected}}
ACTUAL: {{actual}}
ANSWER:
"""


def _get_messages(prompt: str, expected: str, response: str):
    environment = jinja2.Environment()
    template = environment.from_string(EVALUATION_USER_TEMPLATE)
    user_message = template.render({"prompt": prompt, "expected": expected, "actual": response})
    return [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


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
    if not os.environ["OPENAI_API_KEY"]:
        raise PromptToolsUtilityError
    evaluation = openai.ChatCompletion.create(model=model, messages=_get_messages(prompt, expected, response))
    return 1.0 if "RIGHT" in evaluation["choices"][0]["message"]["content"] else 0.0


def evaluate(prompt: str, response: str, metadata: Dict, expected: str) -> float:
    r"""
    Uses auto-evaluation to score the model response.
    """
    return compute(prompt, expected, response)
