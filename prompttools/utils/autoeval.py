import os
import openai
import jinja2
from .error import PromptToolsUtilityError

EVALUATION_SYSTEM_PROMPT = """
Determine whether or not the response is following directions.
Your answer should either be "RIGHT" if the response follows directions,
or "WRONG" if the model is not following directions.
"""

EVALUATION_USER_TEMPLATE = """
PROMPT: {{prompt}}
RESPONSE: {{response}}
ANSWER: 
"""


def _get_messages(prompt: str, response: str):
    environment = jinja2.Environment()
    template = environment.from_string(EVALUATION_USER_TEMPLATE)
    user_message = template.render({"prompt": prompt, "response": response})
    return [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def compute(prompt: str, response: str, model: str = "gpt-4") -> float:
    r"""
    Uses a high quality chat model, like GPT-4, to automatically evaluate a given
    prompt/response pair. Outputs can be 0 or 1.

    Args:
        prompt (str): The input prompt.
        response (str): The model response.
        model (str): The OpenAI chat model to use for generating an expected response.
            Defaults to GPT-4.
    """
    if not os.environ["OPENAI_API_KEY"]:
        raise PromptToolsUtilityError
    evaluation = openai.ChatCompletion.create(
        model=model, messages=_get_messages(prompt, response)
    )
    return 1.0 if "RIGHT" in evaluation["choices"][0]["message"]["content"] else 0.0
