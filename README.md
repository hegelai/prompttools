# prompttools

Welcome to prompttools! This repo offers a set of tools for testing and experimenting with prompts. The core idea is to enable developers to play with prompts using familiar interfaces like _code_ and _notebooks_ instead of a playground.

## Using prompttools

There are primarily two ways you can use prompttools in your LLM workflow:

1. Run experiments in [notebooks](/examples/notebooks/).
1. Write [unit tests](/examples/prompttests/example.py).

## Notebooks

There are a few different ways to run an experiment in a notebook. 

The simplest way is to define an experimentation harness and an evaluation function:

```python
def eval_fn(prompt: str, results: Dict, metadata: Dict) -> float:
    return 1.0

prompt_templates = ["Echo the following input: {{input}}",
                    "Repeat the following input: {{input}}"]

user_inputs = [{"input": "This is a test"}, 
               {"input": "This is not a test"}]

harness = PromptTemplateExperimentationHarness("gpt-3.5-turbo", 
                                               prompt_templates, 
                                               user_inputs)


harness.prepare()
harness.run()
harness.evaluate("metric_name", eval_fn)
harness.visualize()
```

You can also manually enter feedback to evaluate prompts, see [HumanFeedback.ipynb](/examples/notebooks/HumanFeedback.ipynb).

## Unit Tests

Unit tests in `prompttools` are called `prompttests`. They use the `@prompttest` annotation to transform an evaluation function into an efficient unit test. The prompttest framework executes and evaluates experiments so you can test prompts over time. You can see an example test [here](/examples/prompttests/example.py) and an example of that test being used as a Github Action [here](/.github/workflows/post-commit.yaml).

## Persisting Results

To persist the results of your tests and experiments, you can enable `DialecticScribe`, which logs all the inferences from your LLM, along with metadata and custom metrics, for you to view on your [dashboard](https://app.hegel-ai.com).