# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Tuple
import prompttools.prompttest as prompttest
from prompttools.utils import similarity
from prompttools.experiment import HuggingFaceHubExperiment

EXPECTED = {"Who was the first president of the USA?": "George Washington"}

if not (("HUGGINGFACEHUB_API_TOKEN" in os.environ) or ("DEBUG" in os.environ)):
    print(
        "Error: This example requires you to set either your HUGGINGFACEHUB_API_TOKEN or DEBUG=1"
    )
    exit(1)


def extract_responses(output) -> list[str]:
    r"""
    Helper function to unwrap the Hugging Face Hub repsonse object.
    """
    return output["generated_text"]


@prompttest(
    experiment_classname=HuggingFaceHubExperiment,
    model_name="google/flan-t5-xxl",
    metric_name="similar_to_expected",
    prompt_template="Question: {{input}}",
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
    scores = [
        similarity.compute(expected, response, use_chroma=False)
        for response in extract_responses(results)
    ]
    return max(scores)


if __name__ == "__main__":
    prompttest.main()
