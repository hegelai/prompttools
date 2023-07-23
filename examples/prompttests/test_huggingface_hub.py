# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import jinja2
import prompttools.prompttest as prompttest
from prompttools.utils import similarity
from prompttools.experiment import HuggingFaceHubExperiment

if not (("HUGGINGFACEHUB_API_TOKEN" in os.environ) or ("DEBUG" in os.environ)):
    print("Error: This example requires you to set either your HUGGINGFACEHUB_API_TOKEN or DEBUG=1")
    exit(1)


@prompttest.prompttest(
    experiment=HuggingFaceHubExperiment,
    model_name="google/flan-t5-xxl",
    metric_name="similar_to_expected",
    eval_fn=similarity.evaluate,
    expected="George Washington",
)
def prompt_provider():
    prompt_template = "Answer the following question: {{ input }}"
    user_input = {"input": "Who was the first president of the USA?"}
    environment = jinja2.Environment()
    template = environment.from_string(prompt_template)
    prompt = template.render(**user_input)
    return [prompt]


if __name__ == "__main__":
    prompttest.main()
