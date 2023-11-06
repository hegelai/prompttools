# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import pandas.core.series
import jinja2

from prompttools.utils.model_evaluators.EvaluatorUtils import get_evaluator_for_model

try:
    import anthropic
except ImportError:
    anthropic = None


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
    response = get_evaluator_for_model(model).evaluate_and_score(
        model, fact, model_answer
    )
    return int(response)


def autoeval_scoring(row: pandas.core.series.Series, expected: str, response_column_name: str = "response") -> float:
    r"""
    Uses auto-evaluation to score the model response.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        expected (str): the expected response
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    return compute(fact=expected, model_answer=row[response_column_name])
