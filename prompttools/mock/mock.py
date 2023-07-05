# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

r"""
These mock functions exist for testing and demo purposes.
"""

def mock_chat_completion_fn(**kwargs):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The World Series in 2020 was played at Globe Life Field in Arlington, Texas.",
                    "role": "assistant",
                },
            }
        ],
        "created": 1687839008,
        "id": "chatcmpl-7VuJkr4YgPSo2YoFthyq62Kk9SsTY",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {"completion_tokens": 18, "prompt_tokens": 57, "total_tokens": 75},
    }


def mock_completion_fn(**kwargs):
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThe Los Angeles Dodgers won the World Series in 2020",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
