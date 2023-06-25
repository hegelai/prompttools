
import langchain
from typing import Any


# use jinja prompt

class Template:
    def __init__(self):
        pass

    def from_langchain(self, prompt:langchain.prompts.prompt.PromptTemplate) -> 'Template':
        pass

    def from_string(self, prompt: str) -> 'Template':
        pass

    def compose(self, **kwargs: Any) -> str:
        pass

    # def compose_chat(self, **kwargs: Any) -> List:
    #     pass

    # def from_file(path: str) -> 'Template':
    #     pass

    # def from_chat_history(history: List) -> 'Template':
    #     pass

    # def from_examples(examples: List) -> 'Template':
    #     pass

