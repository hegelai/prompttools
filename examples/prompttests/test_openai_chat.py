# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import openai
import jinja2
from prompttools import prompttest
from prompttools.utils import similarity
from prompttools.mock.mock import mock_completion_fn

if not (("OPENAI_API_KEY" in os.environ) or ("DEBUG" in os.environ)):
    print("Error: This example requires you to set either your OPENAI_API_KEY or DEBUG=1")
    exit(1)


def create_prompt():
    prompt_template = "Answer the following question: {{ input }}"
    user_input = {"input": "Who was the first president of the USA?"}
    environment = jinja2.Environment()
    template = environment.from_string(prompt_template)
    return template.render(**user_input)


@prompttest.prompttest(
    metric_name="similar_to_expected",
    eval_fn=similarity.evaluate,
    prompts=[create_prompt()],
    expected=["George Washington"],
)
def completion_fn(prompt: str):
    response = None
    if os.getenv("DEBUG", default=False):
        response = mock_completion_fn(**{"prompt": prompt})
    else:
        response = openai.Completion.create(prompt)
    return response["choices"][0]["text"]


if __name__ == "__main__":
    prompttest.main()
