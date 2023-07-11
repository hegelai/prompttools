from dataclasses import dataclass


@dataclass
class PromptTest:
    "Prompt inputs"
    repo_id: str
    temperature: float
    max_length: int


@dataclass
class Response:
    "Response data returned from LLM"
    repo_id: str
    temperature: float
    max_length: int
    score: float
    response: str