# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

r"""
These mock functions exist for testing and demo purposes.
"""
import json

try:
    import cv2
except ImportError:
    cv2 = None


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


def mock_openai_chat_function_completion_fn(**kwargs):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{\n  "location": "Toronto, Canada",\n  "format": "celsius"\n}',
                    },
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


class _mock_Anthropic_Completion_Object(object):
    def __init__(self, completion: str, model: str, stop_reason: str):
        self.completion = completion
        self.model = model
        self.stop_reason = stop_reason


def mock_anthropic_completion_fn(**kwargs):
    return _mock_Anthropic_Completion_Object(
        completion='{"name": "John", "age": 30, "city": "New York", "interests": ["reading","hiking","coding"]}',
        model="claude-2.0",
        stop_reason="stop_sequence",
    )


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


def mock_mindsdb_completion_fn(**kwargs):
    return [
        (
            "The first president of the United States was George Washington. However, "
            "if you're referring to a different country, please specify so I can provide the correct information.",
        )
    ]


def mock_lc_completion_fn(**kwargs):
    return "The first president of the United States was George Washington."


def mock_stable_diffusion(**kwargs):
    if cv2 is None:
        raise ModuleNotFoundError(
            "Package `cv2` is required to be installed to use this experiment."
            "Please use `pip install opencv-python` to install the package"
        )
    return cv2.imread("/mock_data/images/Just_a_fruit_basket.png")


def mock_replicate_stable_diffusion_completion_fn(model_version: str, **kwargs):
    return ["/mock_data/images/19th_century_wombat_gentleman.png"]


def mock_qdrant_fn(**kwargs):
    from qdrant_client.conversions.common_types import ScoredPoint

    return [
        ScoredPoint(
            id="cf515c6f-1d95-4a21-9052-1a2ecdab34b3",
            version=13,
            score=0.7435235231239,
            payload={"document": "The first president of the United States was George Washington."},
            vector=[0.1, 0.2, 0.3],
        )
    ]
