import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from typing import Dict, List
import prompttools.testing.prompttest as prompttest

# TODO better unit test examples

def extract_responses(output) -> str:
    return [choice["text"] for choice in output["choices"]]


@prompttest.prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="names_dodgers_pt",
    threshold=1,
    prompt_template_file="template.txt",
    user_input_file="user_input.txt",
)
def names_dodgers(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0

@prompttest.prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="names_dodgers_pt",
    threshold=1,
    prompt_template="Answer the following question: {{input}}",
    user_input=[{"input": "Who won the world series in 2020?"}]
)
def names_dodgers_strings(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0


if __name__ == "__main__":
    prompttest.main()
