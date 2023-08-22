## Notebook Examples

In this folder, you will find various examples of how you can `prompttools` for
various experimentation and testing. Often, you can simply change a few parameters
and put in your own test data to make `prompttools` suitable for your use case.

If you have additional use case in mind or spot an issue, please open an issue
and we will be happy to discuss.

We also welcome community contribution of usage examples! Please open a PR if you
have something to share.

### LLM

#### Single Model Examples
- [OpenAI Chat Experiment](OpenAIChatExperiment.ipynb) shows how you can experiment with OpenAI with different models and parameters.
- [OpenAI Chat Function Experiment](OpenAIChatFunctionExperiment.ipynb) shows how you can experiment with OpenAI's function calling API.
- [Anthropic Experiment](AnthropicExperiment.ipynb) shows how you can experiment with Anthropic Claude with different models and parameters.
- [LLaMA Cpp Experiment](LlamaCppExperiment.ipynb) executes LLaMA locally with various parameters and see how it does.
- [LLaMA Cpp Experiment](LlamaCppExperiment.ipynb) executes LLaMA locally with various parameters and see how it does.
- [HuggingFace Hub](HuggingFaceHub.ipynb) compares different OSS models hosted on HuggingFace.
- [GPT-4 Regression](GPT4RegressionTesting.ipynb) examines how the current version GPT-4 model compares with the older, frozen versions.

#### Head To Head Model Comparison

- [Model Comparison](ModelComparison.ipynb) shows how you can compare two OpenAI models.
- [GPT4 vs LLaMA2](GPT4vsLlama2.ipynb) allows you understand if LLaMA might be enough for your use case.
- [LLaMA Head To Head](LlamaHeadToHead.ipynb) presents a match-up between LLaMA 1 and LLaMA 2!

#### Evaluation
- [Auto Evaluation](AutoEval.ipynb) presents an example of how you can use another LLM to evaluate responses.
- [Structured Output](StructuredOutput.ipynb) validates the model outputs adhere to your desired structured format.
- [Semantic Similarity](SemanticSimilarity.ipynb) evaluates your model outputs compared to ideal outputs.
- [Human Feedback](HumanFeedback.ipynb) allows you to provide human feedback to your outputs.


### Vector Databases

- [ChromaDB Experiment](ChromaDBExperiment.ipynb) demonstrates how to experiment with different
  embedding functions and query parameters of `Chroma`. The example evaluates the results by computing the
  ranking correlation against an expected output.
- [Weaviate Experiment](WeaviateExperiment.ipynb) shows how you can easily try different vectorizers, configuration,
  and query functions, and compare the final results.
- [LanceDB Experiment](LanceDBExperiment.ipynb) allows you to try different embedding functions, and query methods.

### Frameworks

- [LangChain Sequential Chain Experiment](LangChainSequentialChainExperiment.ipynb)
- [LangChain Router Chain Experiment](LangChainRouterChainExperiment.ipynb)
- [MindsDB Experiment](MindsDBExperiment.ipynb)

### Computer Vision
- [Stable Diffusion](image_experiments/StableDiffusion.ipynb)
- [Replicate's hosted Stable Diffusion](ReplicateExperiment.ipynb)
