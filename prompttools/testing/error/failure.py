from prompttools.testing.threshold_type import ThresholdType


class PromptTestSetupException(Exception):
    """
    An exception to throw when something goes wrong with the prompt test setup
    """

    pass


def log_failure(metric_name, threshold, actual, threshold_type):
    """
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
