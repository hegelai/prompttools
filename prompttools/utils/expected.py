# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import openai
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
    response = openai.ChatCompletion.create(model=model, prompt=prompt)
    return response["choices"][0]["message"]["content"]


def compute_similarity_against_model(prompt: str, response: str, model: str = "gpt-4") -> str:
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
