# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import pandas.core.series
import jinja2

try:
    import anthropic
except ImportError:
    anthropic = None


AUTO_EVAL_PROMPT_TEMPLATE = """
{{HUMAN_PROMPT}} Given the fact {{fact}}

Evaluate the following Answer on a scale from 1 - 7. Please only respond with an integer from 1 - 7 with no other text.
Lower score means the answer is factually wrong, higher score means the answer is correct. A medium score for
uncertain but not wrong.

Answer: {{model_answer}}

{{AI_PROMPT}}
"""


def _generate_auto_eval_prompt(fact: str, model_answer: str):
    environment = jinja2.Environment()
    template = environment.from_string(AUTO_EVAL_PROMPT_TEMPLATE)
    auto_eval_prompt = template.render(
        {
            "HUMAN_PROMPT": anthropic.HUMAN_PROMPT,
            "AI_PROMPT": anthropic.AI_PROMPT,
            "fact": fact,
            "model_answer": model_answer,
        }
    )
    return auto_eval_prompt


def compute(fact: str, model_answer: str, model: str = "claude-2") -> float:
    r"""
    Uses a high quality chat model, like claude-2, to automatically score a given
    fact/response pair. Output should be an integer ranging from 1 - 7.

    Args:
        fact (str): The fact (truth). The auto-eval model will judge how close the ``response`` is
            from this fact (truth).
        model_answer (str): The model response.
        model (str): The model that will be judging how close is the response from the truth.
            Defaults to Claude 2.
    """
    if not os.environ["ANTHROPIC_API_KEY"]:
        raise RuntimeError("Missing API key for evaluation.")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    completion_response = client.completions.create(
        max_tokens_to_sample=100, model=model, prompt=_generate_auto_eval_prompt(fact, model_answer)
    )
    return int(completion_response.completion)


def autoeval_scoring(row: pandas.core.series.Series, expected: str, response_column_name: str = "response") -> float:
    r"""
    Uses auto-evaluation to score the model response.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        expected (str): the expected response
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    if anthropic is None:
        raise ModuleNotFoundError(
            "Package `anthropic` is required to be installed to use this experiment."
            "Please use `pip install anthropic` to install the package"
        )
    return compute(fact=expected, model_answer=row[response_column_name])
