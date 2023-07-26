from prompttools.experiment import OpenAIChatExperiment
from prompttools.selector.prompt_selector import PromptSelector

PROMPTTOOLS_MD_TMP = "markdown.md"

selectors = [
    PromptSelector("You are a helpful assistant.", "Is 17077 a prime number?"),
    PromptSelector("You are a math tutor.", "Is 17077 a prime number?"),
]
models = ["gpt-3.5-turbo", "gpt-4"]
temperatures = [0.0]
openai_experiment = OpenAIChatExperiment(models, selectors, temperature=temperatures)
openai_experiment.run()

markdown = openai_experiment.to_markdown()
with open(PROMPTTOOLS_MD_TMP, "w") as f:
    f.write(markdown)
