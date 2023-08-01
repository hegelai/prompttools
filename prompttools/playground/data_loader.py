# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import jinja2
import streamlit as st

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.playground.constants import ENVIRONMENT_VARIABLE, EXPERIMENTS


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
def run_multiple(
    model_types,
    models,
    instructions,
    prompts,
    openai_api_key=None,
    anthropic_api_key=None,
    google_api_key=None,
    hf_api_key=None,
):
    import os

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        os.environ["GOOGLE_PALM_API_KEY"] = google_api_key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
    dfs = []
    for i in range(len(models)):
        # TODO Support temperature and other parameters
        selectors = []
        if i + 1 in instructions:
            selectors = [PromptSelector(instructions[i + 1], prompt) for prompt in prompts]
            experiment = EXPERIMENTS[model_types[i]]([models[i]], selectors)
        else:
            experiment = EXPERIMENTS[model_types[i]]([models[i]], prompts)
        dfs.append(experiment.to_pandas_df())
    return dfs
