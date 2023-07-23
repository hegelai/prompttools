# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
import jinja2
from typing import Dict, Tuple
from prompttools import prompttest
from prompttools.utils import similarity
from prompttools.experiment import OpenAICompletionExperiment

if not (("OPENAI_API_KEY" in os.environ) or ("DEBUG" in os.environ)):
    print("Error: This example requires you to set either your OPENAI_API_KEY or DEBUG=1")
    exit(1)

@prompttest.prompttest(
    experiment=OpenAICompletionExperiment,
    model_name="text-davinci-003",
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
