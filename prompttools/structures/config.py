from dataclasses import dataclass
from typing import List


@dataclass
class ModelParams:
    """"
    Testing model requirements.
    """
    hf_repo_ids: List[str]
    temperatures: List[float]
    max_lengths: List[int]
    question: str
    expected: str


MODEL_PARAMS_DEFAULTS = {
    "hf_repo_ids": ["google/flan-t5-xxl", "databricks/dolly-v2-3b", "bigscience/bloom"],
    "temperatures": [0.01, 0.1, 0.5, 1.0],
    "max_lengths": [17, 32, 64, 128],
    "question": "Who was the first president?",
    "expected": "George Washington",
}

def get_model_params_config() -> ModelParams:
    return ModelParams(**dict(MODEL_PARAMS_DEFAULTS))