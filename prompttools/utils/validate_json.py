# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, List, Optional
import pandas.core.series
import json
import re

KEY_EXTRACTION_REGEX = r'"([^"]+?)"\s*:'


def strip_outer_brackets(text: str) -> str:
    r"""
    Removes all chars outside the first '{' and the last '}'. Intended to be a pre-processing
    step prior to parsing a string as JSON.

    Args:
        text(str): the text to process
    """
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    return text[first_brace : last_brace + 1]


def sample_pre_process_fn(text: str):
    r"""
    An example pre-processing that you may use before attempting to parse a string as JSON.
    This function removes all chars outside the first '{' and the last '}'. Then,
    it removes ``"\\n"``.

    This function should be modified depending on your LLM's output.

    Args:
        text(str): the text to process
    """
    text = strip_outer_brackets(text)
    text = text.replace("\\n", "")
    return text


def validate(text: str, pre_process_fn: Optional[Callable] = None):
    r"""
    Validates that the generated text is JSON.

    Args:
        text (str): The generated text, which should be valid JSON.
        pre_process_fn (Callable[str, str]): a function to pre-process the text response from the LLM before attempting
            to parse the string as JSON. Look at ``validate_json.sample_pre_process_fn`` as an example.
    """
    if pre_process_fn:
        text = pre_process_fn(text)
    try:
        json.loads(text)
    except ValueError:
        return 0.0
    return 1.0


def validate_keys(text: str, valid_keys: List[str]):
    r"""
    Guarantees that all keys in the generated JSON are valid.

    Args:
        text (str): The generated text, which should be valid JSON.
        valid_keys (List[str]): A list of valid keys which may appear in the JSON.
    """
    match = re.search(text, KEY_EXTRACTION_REGEX)
    for group in match.groups():
        if group not in valid_keys:
            return 0.0
    return 1.0


def validate_json_response(row: pandas.core.series.Series, response_column_name: str = "response") -> float:
    r"""
    Validate whether ``response`` string is in a valid JSON format.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    return validate(row[response_column_name])


def evaluate(prompt: str, response: str, metadata: Dict) -> float:
    r"""
    Validate whether ``response`` string is in a valid JSON format.

    Args:
        prompt (str): Not used.
        response (str): the string that will be validated
        metadata (dict): Not used.
    """
    return validate(response)
