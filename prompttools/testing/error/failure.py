import logging


class PromptTestFailure(Exception):
    pass


def log_failure(metric_name, threshold, actual):
    print(
        "Test failed:  "
        + metric_name
        + "\nThreshold: "
        + (" " * (len("Test failed") - len("Threshold") + 1))
        + str(threshold)
        + "\nActual: "
        + (" " * (len("Test failed") - len("Actual") + 1))
        + str(actual)
    )
    print("-" * (len("Test failed: " + metric_name) + 2))
