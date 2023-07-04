from enum import Enum


class ThresholdType(Enum):
    """
    Defines the types of thresholds a user can specify for their test case.
    """

    MINIMUM = 1
    MAXIMUM = 2
