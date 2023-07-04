import logging


class PromptTestFailure(Exception):
    pass


def log_failure(metric_name, threshold, actual):
    logging.info(
        "Test failed for metric: "
        + metric_name
        + "\nThreshold: "
        + str(threshold)
        + "\nActual: "
        + str(actual)
    )
