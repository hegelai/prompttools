# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict
import pandas.core.series
from .error import PromptToolsUtilityError

try:
    from pylint import epylint as lint
except ImportError:
    lint = None

PROMPTTOOLS_TMP = "prompttools_tmp.py"


def validate(text: str):
    r"""
    Validates that the generated text is python.

    Args:
        text (str): The generated text, which should be valid python.
    """
    if lint is None:
        raise RuntimeError(
            "Our built-in `validate_python` function requires pylint<3.0. Please use a custom eval function."
            "Feel free to open a GitHub issue or PR."
        )
    if os.path.isfile(PROMPTTOOLS_TMP):
        raise PromptToolsUtilityError
    with open(PROMPTTOOLS_TMP, "w") as f:
        f.write(text)
    pylint_stdout, _ = lint.py_run(PROMPTTOOLS_TMP, return_std=True)
    os.remove(PROMPTTOOLS_TMP)
    return 0.0 if "error" in pylint_stdout.getvalue() else 1.0


def validate_python_response(row: pandas.core.series.Series, response_column_name: str = "response") -> float:
    r"""
    Validate whether ``response`` string follows Python's syntax.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    return validate(row[response_column_name])


def evaluate(prompt: str, response: str, metadata: Dict) -> float:
    r"""
    Validate whether ``response`` string follows Python's syntax.

    Args:
        prompt (str): Not used.
        response (str): the string that will be validated
        metadata (dict): Not used.
    """
    return validate(response)
