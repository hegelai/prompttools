# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import openai
import pandas.core.series
from .error import PromptToolsUtilityError
from . import similarity


def compute(prompt: str, model: str = "gpt-4") -> str:
    r"""
    Computes the expected result of a given prompt by using a high
    quality LLM, like GPT-4.

    Args:
        prompt (str): The input prompt.
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    if not os.environ["OPENAI_API_KEY"]:
        raise PromptToolsUtilityError
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def evaluate(prompt: str, response: str, model: str = "gpt-4") -> str:
    r"""
    Computes the similarity of a given response to the expected result
    generated from a high quality LLM (by default GPT-4) using the same prompt.

    Args:
        prompt (str): The input prompt.
        response (str): The model response.
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    expected_response = compute(prompt, model)
    return similarity.compute(response, expected_response)


def compute_similarity_against_model(
    row: pandas.core.series.Series,
    prompt_column_name: str,
    model: str = "gpt-4",
    response_column_name: str = "response",
) -> str:
    r"""
    Computes the similarity of a given response to the expected result
    generated from a high quality LLM (by default GPT-4) using the same prompt.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        prompt_column_name (str): name of the column that contains the input prompt
        model (str): name of the model that will serve as the judge
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """

    expected_response = compute(row[prompt_column_name], model)
    return similarity.compute(row[response_column_name], expected_response)
