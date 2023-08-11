# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT, AI_PROMPT = None, None


GENERIC_TEMPLATE = """INSTRUCTION:
{instruction}
PROMPT:
{user_input}
RESPONSE:
"""

PALM_TEMPLATE = """{instruction}

{user_input}
"""

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{instruction}
<</SYS>
{user_input} [/INST]
"""

ANTHROPIC_TEMPLATE = """{HUMAN_PROMPT}{instruction}
{user_input}
{AI_PROMPT}"""


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

    def for_anthropic(self):
        return ANTHROPIC_TEMPLATE.format(
            HUMAN_PROMPT=HUMAN_PROMPT, instruction=self.instruction, user_input=self.user_input, AI_PROMPT=AI_PROMPT
        )

    def for_palm(self):
        return PALM_TEMPLATE.format(instruction=self.instruction, user_input=self.user_input)
