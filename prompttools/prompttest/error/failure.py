# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from prompttools.prompttest.threshold_type import ThresholdType


class PromptTestSetupException(Exception):
    r"""
    An exception to throw when something goes wrong with the prompt test setup
    """

    pass


def log_failure(metric_name, threshold, actual, threshold_type):
    r"""
    Prints the test results to the console.
    """
    print(
        "Test failed:  "
        + metric_name
        + "\nThreshold: "
        + (" " * (len("Test failed") - len("Threshold") + 1))
        + str(threshold)
        + "\nActual: "
        + (" " * (len("Test failed") - len("Actual") + 1))
        + str(actual)
        + "\nType: "
        + (" " * (len("Test failed") - len("Type") + 1))
        + str("Minimum" if threshold_type is ThresholdType.MINIMUM else "Maximum")
    )
    print("-" * (len("Test failed: " + metric_name) + 2))
