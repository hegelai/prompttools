import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from typing import Dict, List
from prompttools.testing.decorator import prompttest

# TODO better unit test examples

def extract_responses(output) -> str:
    return [choice["text"] for choice in output["choices"]]


@prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="names_dodgers_pt",
    threshold=1,
    prompt_template_file="template.txt",
    user_input_file="user_input.txt",
    # TODO: Accept model args and string inputs
)
def names_dodgers(
    messages: List[Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0


if __name__ == "__main__":
    names_dodgers()
