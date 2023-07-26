# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import jinja2
import prompttools.prompttest as prompttest
from prompttools.utils import similarity
from prompttools.prompttest.threshold_type import ThresholdType
from prompttools.mock.mock import mock_hf_completion_fn
from huggingface_hub.inference_api import InferenceApi

if not (("HUGGINGFACEHUB_API_TOKEN" in os.environ) or ("DEBUG" in os.environ)):
    print("Error: This example requires you to set either your HUGGINGFACEHUB_API_TOKEN or DEBUG=1")
    exit(1)


client = InferenceApi(
    repo_id="google/flan-t5-xxl",
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    task="text2text-generation",
)


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
    threshold=1.0,
    threshold_type=ThresholdType.MAXIMUM,
)
def completion_fn(prompt: str):
    response = None
    if os.getenv("DEBUG", default=False):
        response = mock_hf_completion_fn(**{"inputs": prompt})
    else:
        response = client(inputs=prompt)
    return response[0]["generated_text"]


if __name__ == "__main__":
    prompttest.main()
