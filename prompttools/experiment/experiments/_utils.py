# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd


def _check_column_uniqueness(column: "pd.core.series.Series") -> bool:
    r"""
    Check if all elements are equal in the column.

    Arg:
        column (pandas.core.series.Series): Column to check
    """
    first_ele = column[0]
    for ele in column:
        if first_ele != ele:
            return True
    return False


def _get_dynamic_columns(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Given a ``pd.DataFrame``, return a DataFrame where columns have more than 1 unique value.

    Args:
        df (pd.DataFrame): DataFrame to examine
    """
    hashable_columns = []
    unhashable_columns = []
    for col in df.columns:
        try:
            hash(df[col][0])
            hashable_columns.append(col)
        except TypeError:
            # If a column is not hashable, check if there exists value differ from the
            if _check_column_uniqueness(df[col]):
                unhashable_columns.append(col)

    unique_counts = df[hashable_columns].nunique()
    columns_with_multiple_unique_values = unique_counts[unique_counts > 1].index
    dfs_to_concat = [df[columns_with_multiple_unique_values], df[unhashable_columns]]
    if (
        "prompt" in df
        and "prompt" not in df[columns_with_multiple_unique_values]
        and "prompt" not in df[unhashable_columns]
    ):
        dfs_to_concat.append(df["prompt"])
    elif (
        "messages" in df
        and "messages" not in df[columns_with_multiple_unique_values]
        and "messages" not in df[unhashable_columns]
    ):
        dfs_to_concat.append(df["messages"])
    return pd.concat(dfs_to_concat, axis=1)
