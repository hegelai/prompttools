# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import jinja2

GENERIC_TEMPLATE = """INSTRUCTION:
{instruction}
PROMPT:
{user_input}
RESPONSE:
"""

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{instruction}
<</SYS>
{user_input} [/INST]
"""


class PromptSelector:
    r"""
    An abstraction for rendering the same prompt
    for different models, e.g. OpenAI Chat models
    and Llama models
    """

    def __init__(self, instruction: str, user_input: object):
        self.instruction = instruction
        self.user_input = user_input

    def for_openai_chat(self):
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.user_input},
        ]

    def for_openai_completion(self):
        return GENERIC_TEMPLATE.format(instruction=self.instruction, user_input=self.user_input)

    def for_huggingface_hub(self):
        return GENERIC_TEMPLATE.format(instruction=self.instruction, user_input=self.user_input)

    def for_llama(self):
        return LLAMA_TEMPLATE.format(instruction=self.instruction, user_input=self.user_input)
