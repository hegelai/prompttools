# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ThresholdType(Enum):
    r"""
    Defines the types of thresholds a user can specify for their test case.
    """

    MINIMUM = 1
    MAXIMUM = 2
