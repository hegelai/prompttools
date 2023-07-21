# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import openai
import logging


def generate_retry_decorator(wait_lower_bound: int = 3, wait_upper_bound: int = 12, max_retry_attempts: int = 5):
    r"""
    Creates a retry decorator that can be used for requests. It looks for specific exceptions and waits for
    certain about of time before retrying. This improves the reliability of the request queue.

    Args:
        wait_lower_bound (int): lower bound to the wait time before retry, defaults to 3.
        wait_upper_bound (int): upper bound to the wait time before retry, defaults to 12.
        max_retry_attempts (int): maximum number of retries before stopping, defaults to 5.
    """
    return retry(
        # For the `i`th attempt, wait 2^i seconds before retrying
        # with lower and upper bound of [3s, 12s].
        wait=wait_exponential(multiplier=1, min=wait_lower_bound, max=wait_upper_bound),
        stop=stop_after_attempt(max_retry_attempts),
        reraise=True,
        retry=(  # Retry for these specific exceptions
            retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
            | retry_if_exception_type(openai.error.Timeout)
        ),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    )


retry_decorator = generate_retry_decorator()
