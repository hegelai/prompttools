# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List
import json
import re

KEY_EXTRACTION_REGEX = r'"([^"]+?)"\s*:'


def validate(text: str):
    r"""
    Validates that the generated text is JSON.

    Args:
        text (str): The generated text, which should be valid JSON.
    """
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


def evaluate(prompt: str, response: str, metadata: Dict) -> float:
    return validate(response)
