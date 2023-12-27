# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import openai
import pandas
from typing import Optional, Union


def apply_moderation(
    row: pandas.core.series.Series,
    text_col_name: str = "response",
    moderation_model: str = "text-moderation-latest",
    category_names: Optional[list[str]] = None,
    category_score_names: Optional[list[str]] = None,
) -> Union[bool, dict]:
    r"""
    Uses OpenAI's moderation API to determine whether the text complies with OpenAI's usage policies.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        text_col_name (str): column name of text to be moderated
        moderation_model (str): name of the OpenAI moderation model, defaults to ``"text-moderation-latest"``
        category_names (Optional[list[str]]): specify the names of category flags to extract from the response and
            be added as column(s) in the row, optional. (e.g. ``["harassment", "violence"]``)
        category_score_names (Optional[list[str]]): specify the names of category scores to extract from the response
            and be added as column(s) in the row, optional. (e.g. ``["harassment", "violence"]``)

    Returns:
        A boolean flag (of whether the input violates policies), or a dict with various topic specific flags/scores.
    """
    text = row[text_col_name]

    moderation_response = openai.moderations.create(model=moderation_model, input=text)
    flagged = moderation_response.results[0].flagged
    res = {}
    if category_names:
        category_flags = moderation_response.results[0].categories.model_dump()
        for c in category_names:
            res[c] = category_flags[c]
    if category_score_names:
        category_scores = moderation_response.results[0].category_scores.model_dump()
        for c in category_score_names:
            res[f"{c}_score"] = category_scores[c]
    if category_names or category_score_names:
        res["moderation_flag"] = flagged
        return res
    else:
        return flagged
