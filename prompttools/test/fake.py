def fake_complete_fn(**kwargs):
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
