from unittest import TestCase

from prompttools.experiment import (
    LlamaCppExperiment,
    HuggingFaceHubExperiment,
    OpenAIChatExperiment,
    OpenAICompletionExperiment,
)


class TestExperiment(TestCase):
    # TODO: Currently, it only ensures importing is correct.
    #       Add unit tests to verify initialization.
    def test_llama_cpp_experiment(self):
        pass

    def test_hugging_face_experiment(self):
        pass

    def test_openai_chat_experiment(self):
        pass

    def test_openai_completion_experiment(self):
        pass
