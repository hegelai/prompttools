# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from prompttools.experiment import LlamaCppExperiment
from prompttools.experiment import OpenAIChatExperiment
from prompttools.experiment import OpenAICompletionExperiment
from prompttools.experiment import AnthropicCompletionExperiment
from prompttools.experiment import GooglePaLMCompletionExperiment
from prompttools.experiment import HuggingFaceHubExperiment
from prompttools.experiment import ReplicateExperiment

ENVIRONMENT_VARIABLE = {
    "Replicate": "REPLICATE_API_TOKEN",
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
    "Replicate": ReplicateExperiment,
}

MODES = ("Instruction", "Prompt Template", "Model Comparison")

MODEL_TYPES = (
    "OpenAI Chat",
    "OpenAI Completion",
    "Anthropic",
    "Google PaLM",
    "LlamaCpp Chat",
    "LlamaCpp Completion",
    "HuggingFace Hub",
    "Replicate",
)

OPENAI_CHAT_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-0314",
    "gpt-4-32k-0314",
)

OPENAI_COMPLETION_MODELS = ("text-davinci-003", "text-davinci-002", "code-davinci-002")
