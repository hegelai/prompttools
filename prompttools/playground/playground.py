# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import streamlit as st
import pyperclip
import urllib.parse

try:
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = Path(base_dir)
    repo_dir = path.parent.parent.absolute()
    load_dotenv(os.path.join(repo_dir, ".env"))
except Exception:
    pass

from prompttools.playground.constants import MODES, MODEL_TYPES, OPENAI_CHAT_MODELS, OPENAI_COMPLETION_MODELS
from prompttools.playground.data_loader import render_prompts, load_data, run_multiple


params = st.experimental_get_query_params()
st.experimental_set_query_params()

st.header("PromptTools Playground")
st.write("Give us a \U00002B50 on [GitHub](https://github.com/hegelai/prompttools)")

with st.sidebar:
    mode = st.radio("Choose a mode", MODES, index=MODES.index(params["mode"][0]) if "mode" in params else 0)
    if mode != "Model Comparison":
        model_type = st.selectbox(
            "Model Type", MODEL_TYPES, index=MODEL_TYPES.index(params["model_type"][0]) if "model_type" in params else 0
        )
        model, api_key = None, None
        if model_type in {"LlamaCpp Chat", "LlamaCpp Completion"}:
            model = st.text_input("Local Model Path", key="llama_cpp_model_path")
        elif model_type == "HuggingFace Hub":
            model = st.text_input("Repo ID", key="hf_model_id")
            api_key = st.text_input("HuggingFace Hub API Key")
        elif model_type == "Google PaLM":
            model = st.text_input("Model", key="palm_model")
            api_key = st.text_input("Google PaLM API Key")
        elif model_type == "Anthropic":
            model = st.selectbox("Model", ("claude-2", "claude-instant-1"))
            api_key = st.text_input("Anthropic API Key")
        elif model_type == "OpenAI Chat":
            model = st.selectbox(
                "Model",
                OPENAI_CHAT_MODELS,
                index=OPENAI_CHAT_MODELS.index(params["model"][0]) if "model" in params else 0,
            )
            api_key = st.text_input("OpenAI API Key")
        elif model_type == "OpenAI Completion":
            model = st.selectbox(
                "Model",
                OPENAI_COMPLETION_MODELS,
                index=OPENAI_COMPLETION_MODELS.index(params["model"][0]) if "model" in params else 0,
            )
            api_key = st.text_input("OpenAI API Key")

        variable_count = 0
        if mode == "Prompt Template":
            instruction_count = st.number_input("Add Template", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add User Input", step=1, min_value=1, max_value=10)
            variable_count = len(params["var_names"][0].split(",")) if "var_names" in params else 1
            variable_count = st.number_input("Add Variable", step=1, min_value=1, max_value=10, value=variable_count)
        elif model_type == "OpenAI Chat":
            instruction_count = st.number_input("Add System Message", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add User Message", step=1, min_value=1, max_value=10)

        else:
            instruction_count = st.number_input("Add Instruction", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)
        var_names = []
        for i in range(variable_count):
            var_names.append(
                st.text_input(
                    f"Variable {i+1} Name",
                    value=params["var_names"][0].split(",")[i] if "var_names" in params else f"Variable {i+1}",
                    key=f"varname_{i}",
                )
            )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperature")
        top_p = None
        max_tokens = None
        presence_penalty = None
        frequency_penalty = None
        if model_type == "OpenAI Chat" or model_type == "OpenAI Completion":
            # top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="top_p")
            # max_tokens = st.number_input("Max Tokens", min_value=0, value=, step=1, key="max_tokens")
            presence_penalty = st.slider(
                "Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key="presence_penalty"
            )
            frequency_penalty = st.slider(
                "Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key="frequency_penalty"
            )
    else:
        model_count = st.number_input("Add Model", step=1, min_value=1, max_value=5)
        prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)
        openai_api_key = st.text_input("OpenAI API Key")
        anthropic_api_key = st.text_input("Anthropic Key")
        google_api_key = st.text_input("Google PaLM Key")
        hf_api_key = st.text_input("HuggingFace Hub Key")


if mode == "Instruction":
    placeholders = [[st.empty() for _ in range(instruction_count + 1)] for _ in range(prompt_count)]

    cols = st.columns(instruction_count + 1)

    with cols[0]:
        a = None
    instructions = []
    for j in range(1, instruction_count + 1):
        with cols[j]:
            instructions.append(
                st.text_area(
                    "System Message" if model_type == "OpenAI Chat" else "Instruction",
                    value=params["instruction"][0] if "instruction" in params else "You are a helpful AI assistant.",
                    key=f"col_{j}",
                )
            )

    prompts = []
    for i in range(prompt_count):
        cols = st.columns(instruction_count + 1)
        with cols[0]:
            prompts.append(
                st.text_area(
                    "User Message" if model_type == "OpenAI Chat" else "Prompt",
                    key=f"row_{i}",
                    value=params["prompt"][0] if "prompt" in params else "",
                )
            )
        for j in range(1, instruction_count + 1):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    run_button, clear_button, share_button = st.columns([1, 1, 1], gap="small")
    with run_button:
        run = st.button("Run")
    with clear_button:
        clear = st.button("Clear")
    with share_button:
        share = st.button("Share")

    link = "https://prompttools.streamlit.app?"
    link += "mode=" + urllib.parse.quote(mode)
    link += "&model_type=" + urllib.parse.quote(model_type)
    link += "&model=" + urllib.parse.quote(model)
    link += "&instruction=" + urllib.parse.quote(instructions[0])
    link += "&prompt=" + urllib.parse.quote(prompts[0])

    if run:
        df = load_data(
            model_type,
            model,
            instructions,
            prompts,
            temperature,
            top_p,
            max_tokens,
            frequency_penalty,
            presence_penalty,
            api_key,
        )
        st.session_state.df = df
        for i in range(len(prompts)):
            for j in range(len(instructions)):
                placeholders[i][j + 1].markdown(df["response"][i + len(prompts) * j])
    elif "df" in st.session_state and not clear:
        df = st.session_state.df
        for i in range(len(prompts)):
            for j in range(len(instructions)):
                placeholders[i][j + 1].markdown(df["response"][i + len(prompts) * j])
elif mode == "Prompt Template":
    instruction = None
    if model_type == "LlamaCpp Chat":
        instruction = st.text_area(
            "Instruction",
            key="instruction",
            value=params["instruction"][0] if "instruction" in params else "You are a helpful AI assistant.",
        )
    elif model_type == "OpenAI Chat":
        instruction = st.text_area(
            "System Message",
            key="instruction",
            value=params["instruction"][0] if "instruction" in params else "You are a helpful AI assistant.",
        )

    placeholders = [[st.empty() for _ in range(instruction_count + variable_count)] for _ in range(prompt_count)]

    cols = st.columns(instruction_count + variable_count)

    with cols[0]:
        a = None
    templates = []
    for j in range(variable_count, instruction_count + variable_count):
        with cols[j]:
            templates.append(
                st.text_area(
                    "Prompt Template",
                    key=f"col_{j-variable_count}",
                    value=params["template"][0] if "template" in params else "",
                )
            )

    vars = []
    varlist = []
    if "vars" in params:
        varlist = params["vars"][0].split(",")
    for i in range(prompt_count):
        cols = st.columns(instruction_count + variable_count)
        vars.append(dict())
        for j in range(variable_count):
            with cols[j]:
                vars[i][var_names[j]] = st.text_area(
                    var_names[j], key=f"var_{i}_{j}", value=varlist[j] if len(varlist) > 0 else ""
                )
        for j in range(variable_count, instruction_count + variable_count):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    run_button, clear_button, share_button = st.columns([1, 1, 1], gap="small")
    with run_button:
        run = st.button("Run")
    with clear_button:
        clear = st.button("Clear")
    with share_button:
        share = st.button("Share")

    link = "https://prompttools.streamlit.app?"
    link += "mode=" + urllib.parse.quote(mode)
    link += "&model_type=" + urllib.parse.quote(model_type)
    link += "&model=" + urllib.parse.quote(model)
    if instruction:
        link += "&instruction=" + urllib.parse.quote(instruction)
    link += "&template=" + urllib.parse.quote(templates[0])
    if len(var_names) > 0:
        link += "&var_names=" + urllib.parse.quote(",".join(var_names))
    if len(vars) > 0:
        link += "&vars=" + urllib.parse.quote(",".join(vars[0].values()))

    if run:
        prompts = render_prompts(templates, vars)
        df = load_data(model_type, model, [instruction], prompts, temperature, api_key)
        st.session_state.prompts = prompts
        st.session_state.df = df
        for i in range(len(prompts)):
            for j in range(len(templates)):
                placeholders[i][j + variable_count].markdown(df["response"][i + len(prompts) * j])
    elif "df" in st.session_state and "prompts" in st.session_state and not clear:
        df = st.session_state.df
        prompts = st.session_state.prompts
        for i in range(len(prompts)):
            for j in range(len(templates)):
                placeholders[i][j + variable_count].markdown(df["response"][i + len(prompts) * j])
elif mode == "Model Comparison":
    placeholders = [[st.empty() for _ in range(model_count + 1)] for _ in range(prompt_count)]

    cols = st.columns(model_count + 1)

    with cols[0]:
        a = None
    models = []
    model_types = []
    instructions = {}
    for j in range(1, model_count + 1):
        with cols[j]:
            model_types.append(
                st.selectbox(
                    "Model Type",
                    (
                        "OpenAI Chat",
                        "OpenAI Completion",
                        "Anthropic",
                        "Google PaLM",
                        "LlamaCpp Chat",
                        "LlamaCpp Completion",
                        "HuggingFace Hub",
                    ),
                    key=f"type_{j}",
                )
            )
            model = None
            if model_types[j - 1] == "LlamaCpp Chat":
                models.append(st.text_input("Local Model Path", key=f"path_{j}"))
                instructions[j] = st.text_area("Instruction", key=f"instruction_{j}")
            elif model_types[j - 1] == "LlamaCpp Completion":
                models.append(st.text_input("Local Model Path", key=f"path_{j}"))
            elif model_types[j - 1] == "HuggingFace Hub":
                models.append(st.text_input("Repo ID", key=f"model_id_{j}"))
            elif model_types[j - 1] == "Google PaLM":
                models.append(st.text_input("Model", key=f"palm_model_{j}"))
            elif model_types[j - 1] == "Anthropic":
                models.append(st.selectbox("Model", ("claude-2", "claude-instant-1"), key=f"anthropic_{j}"))
            elif model_types[j - 1] == "OpenAI Chat":
                models.append(
                    st.selectbox(
                        "Model",
                        (
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
                        ),
                        key=f"model_{j}",
                    )
                )
                instructions[j] = st.text_area(
                    "System Message", value="You are a helpful AI assistant.", key=f"instruction_{j}"
                )

    prompts = []
    for i in range(prompt_count):
        cols = st.columns(model_count + 1)
        with cols[0]:
            prompts.append(st.text_area("Prompt", key=f"row_{i}"))
        for j in range(1, model_count + 1):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    run_button, clear_button, share_button = st.columns([1, 1, 1], gap="small")
    with run_button:
        run = st.button("Run")
    with clear_button:
        clear = st.button("Clear")
    with share_button:
        share = st.button("Share")

    link = "https://prompttools.streamlit.app?"
    link += "mode=" + urllib.parse.quote(mode)
    link += "&model_type=" + urllib.parse.quote(model_type)
    link += "&model=" + urllib.parse.quote(model)
    link += "&instruction=" + urllib.parse.quote(instructions[0])
    link += "&prompt=" + urllib.parse.quote(prompts[0])

    if run:
        dfs = run_multiple(
            model_types, models, instructions, prompts, openai_api_key, anthropic_api_key, google_api_key, hf_api_key
        )
        st.session_state.dfs = dfs
        for i in range(len(prompts)):
            for j in range(len(models)):
                placeholders[i][j + 1].markdown(dfs[j]["response"][i])

    elif "dfs" in st.session_state and not clear:
        dfs = st.session_state.dfs
        for i in range(len(prompts)):
            for j in range(len(models)):
                placeholders[i][j + 1].markdown(dfs[j]["response"][i])

if clear:
    for key in st.session_state.keys():
        del st.session_state[key]


if share:
    pyperclip.copy(link)
