# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import openai
import jinja2
from prompttools import prompttest
from prompttools.prompttest.threshold_type import ThresholdType
from prompttools.utils import similarity
from prompttools.utils import validate_json
from prompttools.mock.mock import mock_openai_completion_fn


if not (("OPENAI_API_KEY" in os.environ) or ("DEBUG" in os.environ)):
    print("Error: This example requires you to set either your OPENAI_API_KEY or DEBUG=1")
    exit(1)


def create_json_prompt():
    prompt_template = "Answer the following question using a valid JSON format: {{ input }}"
    user_input = {"input": "Who was the first president?"}
    environment = jinja2.Environment()
    template = environment.from_string(prompt_template)
    return template.render(**user_input)


def create_prompt():
    prompt_template = "Answer the following question: {{ input }}"
    user_input = {"input": "Who was the first president of the USA?"}
    environment = jinja2.Environment()
    template = environment.from_string(prompt_template)
    return template.render(**user_input)


@prompttest.prompttest(
    metric_name="is_valid_json",
    eval_fn=validate_json.evaluate,
    prompts=[create_json_prompt()],
)
def json_completion_fn(prompt: str):
    if os.getenv("DEBUG", default=False):
        response = mock_openai_completion_fn(**{"prompt": prompt})
    else:
        response = openai.completions.create(model="babbage-002", prompt=prompt)
    return response.choices[0].text


@prompttest.prompttest(
    metric_name="similar_to_expected",
    eval_fn=similarity.evaluate,
    prompts=[create_prompt()],
    expected=["George Washington"],
    threshold=1.0,
    threshold_type=ThresholdType.MAXIMUM,
)
def completion_fn(prompt: str):
    if os.getenv("DEBUG", default=False):
        response = mock_openai_completion_fn(**{"prompt": prompt})
    else:
        response = openai.completions.create(model="babbage-002", prompt=prompt)
    return response.choices[0].text


if __name__ == "__main__":
    prompttest.main()
