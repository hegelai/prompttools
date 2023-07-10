from dataclasses import dataclass


@dataclass
class PromptTest:
    "Prompt inputs"
    repo_id: str
    temperature: float
    max_length: int
