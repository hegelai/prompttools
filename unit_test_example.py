from typing import Dict, List
from prompttools.testing.decorator import prompttest


def extract_responses_pt(output) -> str:
    return [choice["text"] for choice in output["choices"]]


def extract_responses(output) -> str:
    return [choice["message"]["content"] for choice in output["choices"]]


@prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="names_dodgers",
    threshold=1,
    system_prompt_file="prompt.txt",
    human_messages_file="messages.txt",
)
def names_dodgers(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0


@prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="names_dodgers_pt",
    threshold=1,
    prompt_template_file="template.txt",
    user_input_file="user_input.txt",
)
def names_dodgers_pt(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    responses = extract_responses_pt(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0


names_dodgers_pt()
names_dodgers()
