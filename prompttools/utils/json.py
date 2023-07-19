from typing import List
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
