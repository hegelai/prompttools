import os
import jinja2
import streamlit as st

from prompttools.selector.prompt_selector import PromptSelector
from ui.constants import ENVIRONMENT_VARIABLE, EXPERIMENTS


def render_prompts(templates, vars):
    prompts = []
    for i, template in enumerate(templates):
        environment = jinja2.Environment()
        jinja_template = environment.from_string(template)
        prompts.append(jinja_template.render(**vars[i]))
    return prompts


@st.cache_data
def load_data(model_type, model, instructions, user_inputs, temperature=0.0, api_key=None):
    if api_key:
        os.environ[ENVIRONMENT_VARIABLE[model_type]] = api_key
    selectors = [PromptSelector(instruction, user_input) for instruction in instructions for user_input in user_inputs]

    experiment = None
    if model_type == "LlamaCpp Chat":
        call_params = dict(temperature=[temperature])
        experiment = EXPERIMENTS[model_type]([model], selectors, call_params=call_params)
    elif model_type in {"OpenAI Chat", "OpenAI Completion"}:
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    elif model_type == "HuggingFace Hub":
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    return experiment.to_pandas_df()


@st.cache_data
def run_multiple():
    pass
