from typing import Dict, List
from prompttools.experiment.openai_completion_experiment import (
    OpenAICompletionExperiment,
)
from prompttools.harness.prompt_template_harness import (
    PromptTemplateExperimentationHarness,
)


def extract_responses(output) -> str:
    return [choice["text"] for choice in output["choices"]]


# Define a list of chat histories over which to run your experiment
prompt_templates = ["Answer the following question: {{input}}"]
user_inputs = [{"input": "Who won the world series in 2020?"}]


# Define an evaluation function that assigns scores to each inference
def eval_fn(messages: List[Dict[str, str]], results: Dict, metadata: Dict) -> float:
    responses = extract_responses(results)
    for response in responses:
        if "Dodgers" in response:
            return 1.0
    return 0.0


# Define an experimentation harness using the class name for the underlying experiment
harness = PromptTemplateExperimentationHarness(
    OpenAICompletionExperiment, "gpt-3.5-turbo", prompt_templates, user_inputs
)

# Run the evaluation
harness.prepare()
harness.run()
harness.evaluate("names_dodgers", eval_fn)
harness.visualize(pivot=True)
