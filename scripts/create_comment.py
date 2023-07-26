from prompttools.experiment import OpenAIChatExperiment

PROMPTTOOLS_MD_TMP = "markdown.md"

prompts = ["Tell me a joke.", "Is 17077 a prime number?"]
models = ["gpt-3.5-turbo", "gpt-4"]
temperatures = [0.0]
openai_experiment = OpenAIChatExperiment(models, prompts, temperature=temperatures)
openai_experiment.run()

markdown = openai_experiment.to_markdown()
with open(PROMPTTOOLS_MD_TMP, "w") as f:
    f.write(markdown)