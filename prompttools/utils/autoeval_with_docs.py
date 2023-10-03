# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import openai
import pandas.core.series
import jinja2
from .error import PromptToolsUtilityError


EVALUATION_SYSTEM_PROMPT = """
Using the provided documents, determine whether or not the response is accurate.
Your answer should be an integer rating from 0 to 10, with 0 being extremely inaccurate
and 10 being perfectly accurate. Only an integer should be returned in the response.
"""

EVALUATION_USER_TEMPLATE = """
DOCUMENTS:
{{documents}}

RESPONSE: {{response}}
ANSWER:
"""


def _get_messages(documents: list[str], response: str):
    environment = jinja2.Environment()
    template = environment.from_string(EVALUATION_USER_TEMPLATE)
    user_message = template.render({"documents": "\n".join(documents), "response": response})
    return [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def compute(documents: list[str], response: str, model: str = "gpt-4") -> float:
    r"""
    Uses a high quality chat model, like GPT-4, to automatically evaluate a given
    prompt/response pair. Outputs can be 0 or 1.

    Args:
        documents (list[str]): documents to provide relevant context for the model to judge
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    if not os.environ["OPENAI_API_KEY"]:
        raise PromptToolsUtilityError
    evaluation = openai.ChatCompletion.create(model=model, messages=_get_messages(documents, response))
    score_text = evaluation["choices"][0]["message"]["content"]
    return int(score_text)


def autoeval_with_documents(
    row: pandas.core.series.Series,
    documents: list[str],
    response_column_name: str = "response",
) -> float:
    r"""
    Given a list of documents, score whether the model response is accurate with "gpt-4" as the judge,
    returning an integer score from 0 to 10.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        documents (list[str]): documents to provide relevant context for the model to judge
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    return compute(documents, row[response_column_name])
