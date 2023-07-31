from prompttools.experiment import LlamaCppExperiment
from prompttools.experiment import OpenAIChatExperiment
from prompttools.experiment import OpenAICompletionExperiment
from prompttools.experiment import AnthropicCompletionExperiment
from prompttools.experiment import GooglePaLMCompletionExperiment
from prompttools.experiment import HuggingFaceHubExperiment

ENVIRONMENT_VARIABLE = {
    "OpenAI Chat": "OPENAI_API_KEY",
    "OpenAI Completion": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google PaLM": "GOOGLE_PALM_API_KEY",
    "HuggingFace Hub": "HUGGINGFACEHUB_API_TOKEN",
}

EXPERIMENTS = {
    "LlamaCpp Chat": LlamaCppExperiment,
    "OpenAI Chat": OpenAIChatExperiment,
    "OpenAI Completion": OpenAICompletionExperiment,
    "Anthropic": AnthropicCompletionExperiment,
    "Google PaLM": GooglePaLMCompletionExperiment,
    "HuggingFace Hub": HuggingFaceHubExperiment,
}
