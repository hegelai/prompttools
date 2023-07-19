# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Tuple
from prompttools import prompttest
from prompttools.utils import similarity
from prompttools.experiment import OpenAIChatExperiment

EXPECTED = {"Who was the first president of the USA?": "George Washington"}

if not (("OPENAI_API_KEY" in os.environ) or ("DEBUG" in os.environ)):
    print(
        "Error: This example requires you to set either your OPENAI_API_KEY or DEBUG=1"
    )
    exit(1)


def extract_responses(output) -> list[str]:
    r"""
    Helper function to unwrap the OpenAI repsonse object.
    """
    return [choice["message"]["content"] for choice in output["choices"]]


@prompttest.prompttest(
    experiment=OpenAIChatExperiment,
    model_name="text-davinci-003",
    metric_name="similar_to_expected",
    prompt_template="Answer the following question: {{input}}",
    user_input=[{"input": "Who was the first president of the USA?"}],
)
def measure_similarity(
    input_pair: Tuple[str, Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    r"""
    A simple test that checks semantic similarity between the user input
    and the model's text responses.
    """
    expected = EXPECTED[input_pair[1]["input"]]
    distances = [
        similarity.compute(expected, response)
        for response in extract_responses(results)
    ]
    return min(distances)


if __name__ == "__main__":
    prompttest.main()
