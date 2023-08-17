# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import scipy.stats as stats
import pandas


def ranking_correlation(
    row: pandas.core.series.Series, expected_ranking: list, ranking_column_name: str = "top doc ids"
) -> float:
    r"""
    A simple test that compares the expected ranking for a given query with the actual ranking produced
    by the embedding function being tested.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        expected_ranking (list): the expected list of ranking to compare
        ranking_column_name (str): the column name of the actual ranking produced by the model,
            defaults to ``"top doc ids"``

    Example:
        >>> EXPECTED_RANKING_LIST = [
        >>>     ["id1", "id3", "id2"],
        >>>     ["id2", "id3", "id1"],
        >>>     ["id1", "id3", "id2"],
        >>>     ["id2", "id3", "id1"],
        >>> ]
        >>> experiment.evaluate("ranking_correlation", ranking_correlation, expected_ranking=EXPECTED_RANKING_LIST)
    """
    correlation, _ = stats.spearmanr(row[ranking_column_name], expected_ranking)
    return correlation
