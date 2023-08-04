# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict
<<<<<<< HEAD
import openai
import pandas.core.series
=======
>>>>>>> 862ca6e (add framework for adding new model evaluators)
import jinja2

from .model_evaluators.EvaluatorUtils import get_evaluator_for_model

EVALUATION_SYSTEM_PROMPT = """
Determine whether or not the response is following directions.
Your answer should either be "RIGHT" if the response follows directions,
or "WRONG" if the model is not following directions.
"""

EVALUATION_USER_TEMPLATE = """
PROMPT: {{prompt}}
RESPONSE: {{response}}
"""


def _get_user_prompt(prompt: str, response: str):
    environment = jinja2.Environment()
    template = environment.from_string(EVALUATION_USER_TEMPLATE)
    user_prompt = template.render({"prompt": prompt, "response": response})
    return user_prompt


def compute(prompt: str, response: str, model: str = "gpt-4") -> float:
    r"""
    Uses a high quality chat model, like GPT-4, to automatically evaluate a given
    prompt/response pair. Outputs can be 0 or 1.

    Args:
        prompt (str): The input prompt.
        response (str): The model response.
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    response = get_evaluator_for_model(model).evaluate(model, EVALUATION_SYSTEM_PROMPT, _get_user_prompt())
    return 1.0 if "RIGHT" in response else 0.0


def evaluate(prompt: str, response: str, _metadata: Dict) -> float:
    r"""
    Uses auto-evaluation to score the model response with "gpt-4" as the judge, returning 0.0 or 1.0.

    Args:
        prompt (str): The input prompt.
        response (str): The model response.
        metadata (str): Not used.
    """
    return compute(prompt, response)


def autoeval_binary_scoring(
    row: pandas.core.series.Series,
    prompt_column_name: str,
    response_column_name: str = "response",
) -> float:
    r"""
    Uses auto-evaluation to score the model response with "gpt-4" as the judge, returning 0.0 or 1.0.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        prompt_column_name (str): name of the column that contains the input prompt
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    return compute(row[prompt_column_name], row[response_column_name])
