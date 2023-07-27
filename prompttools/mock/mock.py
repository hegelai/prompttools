# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

r"""
These mock functions exist for testing and demo purposes.
"""
import json


def mock_openai_chat_completion_fn(**kwargs):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "George Washington",
                    "role": "assistant",
                },
            }
        ],
        "created": 1687839008,
        "id": "",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {"completion_tokens": 18, "prompt_tokens": 57, "total_tokens": 75},
    }


def mock_openai_completion_fn(**kwargs):
    return {
        "id": "",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": json.dumps({"text": "George Washington"}),
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def mock_hf_completion_fn(**kwargs):
    return [{"generated_text": "George Washington"}]


def mock_chromadb_fn(**kwargs):
    return {
        "ids": [["id1"]],
        "embeddings": None,
        "documents": [["George Washington lived in modern day Philadelphia"]],
        "metadatas": [[{"source": "my_source"}]],
        "distances": [[0.5932742953300476]],
    }


def mock_anthropic_completion_fn(**kwargs):
    return {
        "id": "cmpl-6L7J9GfXgcYBEZa8zcjj1oJrqtX",
        "object": "conversation",
        "created": 1674647865,
        "model": "claude-standard-v1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "I'm doing well, thanks for asking!"}}],
    }


class _mock_PaLM_Completion_Object(object):
    def __init__(self, candidates: list[dict], result: str, filters: list = [], safety_feedback: list = []):
        self.candidates = candidates
        self.result = result
        self.filters = filters
        self.safety_feedback = safety_feedback


def mock_palm_completion_fn(**kwargs):
    return _mock_PaLM_Completion_Object(
        candidates=[{"output": "How are you today?", "safety_ratings": []}], result="How are you today?"
    )
