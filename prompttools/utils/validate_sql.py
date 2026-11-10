# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import pandas.core.series


def validate(text: str) -> float:
    """Score a SELECT query using sqlvalidator's supported validation rules.

    Install the optional dependency with ``pip install 'prompttools[sql]'``.
    This checks SQL text without executing it or connecting to a database.
    The parser supports only SELECT statements and a limited set of checks;
    a passing score does not establish dialect, schema, or execution correctness.

    Args:
        text (str): The SQL response to validate.
    """
    try:
        import sqlvalidator
    except ImportError as exc:
        raise ModuleNotFoundError("SQL validation requires `pip install 'prompttools[sql]'`.") from exc

    try:
        return 1.0 if sqlvalidator.parse(text).is_valid() else 0.0
    except StopIteration:
        # sqlvalidator 0.0.20 raises this for truncated input such as SELECT * FROM.
        return 0.0


def validate_sql_response(row: pandas.core.series.Series, response_column_name: str = "response") -> float:
    """Evaluate SQL from an experiment result row without executing the query.

    Args:
        row (pandas.core.series.Series): A row of experiment inputs, responses, and metrics.
        response_column_name (str): The column containing SQL, defaults to ``response``.
    """
    return validate(row[response_column_name])


def evaluate(prompt: str, response: str, metadata: Dict) -> float:
    """Validate a response through the prompttest evaluation interface.

    ``prompt`` and ``metadata`` are unused. See :func:`validate` for the parser's
    supported checks and limitations.
    """
    return validate(response)
