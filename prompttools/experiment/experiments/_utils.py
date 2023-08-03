# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd


def _get_dynamic_columns(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Given a ``pd.DataFrame``

    Args:
        df (pd.DataFrame):
    """
    hashable_columns = []
    unhashable_columns = []
    for col in df.columns:
        try:
            hash(df[col][0])
            hashable_columns.append(col)
        except TypeError:
            unhashable_columns.append(col)

    unique_counts = df[hashable_columns].nunique()
    columns_with_multiple_unique_values = unique_counts[unique_counts > 1].index
    return pd.concat([df[columns_with_multiple_unique_values], df[unhashable_columns]], axis=1)
