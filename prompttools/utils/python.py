import os
from .error import PromptToolsUtilityError
from pylint import epylint as lint
import logging

PROMPTTOOLS_TMP = "prompttools_tmp.py"


def validate(text: str):
    r"""
    Validates that the generated text is python.

    Args:
        text (str): The generated text, which should be valid python.
    """
    if os.path.isfile(PROMPTTOOLS_TMP):
        raise PromptToolsUtilityError
    with open(PROMPTTOOLS_TMP, "w") as f:
        f.write(text)
    pylint_stdout, _ = lint.py_run(PROMPTTOOLS_TMP, return_std=True)
    os.remove(PROMPTTOOLS_TMP)
    return 0.0 if "error" in pylint_stdout.getvalue() else 1.0
