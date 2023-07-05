# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import chromadb
from typing import Dict, Tuple
import prompttools.testing.prompttest as prompttest
from prompttools.utils import similarity

EXPECTED = {"Who was the first president of the USA?": "George Washington"}


def extract_responses(output) -> str:
    r"""
    Helper function to unwrap the OpenAI repsonse object.
    """
    return [choice["text"] for choice in output["choices"]]


@prompttest.prompttest(
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
