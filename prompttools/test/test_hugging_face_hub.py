# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from prompttools.experiment import HuggingFaceHubExperiment
from prompttools.utils import similarity
from typing import Dict, List


def test_single_model() -> None:
    "Test one model"
    hf_repo_ids = ["google/flan-t5-xxl"]
    max_lengths = [32]
    temperatures = [0.1]

    prompts = ["Who was the very first president of the USA?"]

    tasks = ["text-generation", "summarization"]

    model_kwargs = {"model_kwargs":
        {"max_length": max_lengths,
        "temperature": temperatures,}
    }

    experiment = HuggingFaceHubExperiment(
        repo_id=hf_repo_ids,
        prompt=prompts,
        task=tasks,
        kwargs=model_kwargs,
    )

    experiment.run()

    experiment.evaluate("similar_to_expected", measure_similarity)

    for score in experiment.scores["similar_to_expected"]:
        assert score > 0.30
    
    total_combos = len(hf_repo_ids) * len(max_lengths) * len(temperatures) * len(prompts) * len(tasks)
    assert len(experiment.results) == total_combos

def test_multi_models() -> None:
    "Test multiple models"
    hf_repo_ids = ["google/flan-t5-xxl", "bigscience/bloom"]
    max_lengths = [32]
    temperatures = [0.1]

    prompts = ["Who was the very first president of the USA?"]

    tasks = ["text-generation", "summarization"]

    model_kwargs = {"model_kwargs":
        {"max_length": max_lengths,
        "temperature": temperatures,}
    }

    experiment = HuggingFaceHubExperiment(
        repo_id=hf_repo_ids,
        prompt=prompts,
        task=tasks,
        kwargs=model_kwargs,
    )

    experiment.run()

    experiment.evaluate("similar_to_expected", measure_similarity)

    for score in experiment.scores["similar_to_expected"]:
        assert score > 0.30
    
    total_combos = len(hf_repo_ids) * len(max_lengths) * len(temperatures) * len(prompts) * len(tasks)
    assert len(experiment.results) == total_combos


def extract_responses(output, task_type: str) -> List[str]:
    return [resp[task_type] for resp in output]


def measure_similarity(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    """
    A simple test that checks semantic similarity between the user input
    and the model's text responses.
    """
    scores = [
        similarity.compute("George Wash", response, use_chroma=False)
        for response in extract_responses(results, "generated_text")
    ]
    return max(scores)