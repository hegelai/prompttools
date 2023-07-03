from typing import Dict, List
from prompttools.experiment.openai_chat_experiment import OpenAIChatExperiment
from prompttools.harness.chat_history_harness import ChatHistoryExperimentationHarness


def extract_responses(output) -> list:
    return [choice["message"]["content"] for choice in output["choices"]]


# Define a list of chat histories over which to run your experiment
chat_histories: List[List[Dict[str, str]]] = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ]
]


# Define an evaluation function that assigns scores to each inference
def eval_fn(messages: List[Dict[str, str]], results: Dict, metadata: Dict) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Arlington" in response:
            return 1.0
    return 0.0


# Define an experimentation harness using the class name for the underlying experiment
harness = ChatHistoryExperimentationHarness(
    OpenAIChatExperiment, "gpt-3.5-turbo", chat_histories
)

# Run the evaluation
harness.prepare()
harness.run()
harness.evaluate("names_arlington", eval_fn)
harness.visualize()
