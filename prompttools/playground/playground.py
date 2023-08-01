# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import streamlit as st

from prompttools.playground.data_loader import render_prompts, load_data, run_multiple

st.header("PromptTools Playground")

with st.sidebar:
    mode = st.radio("Choose a mode", ("Instruction", "Prompt Template", "Model Comparison"))

    if mode != "Model Comparison":
        model_type = st.selectbox(
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
            )
            api_key = st.text_input("OpenAI API Key")
        elif model_type == "OpenAI Completion":
            model = st.selectbox("Model", ("text-davinci-003", "text-davinci-002", "code-davinci-002"))
            api_key = st.text_input("OpenAI API Key")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="temperature")
        variable_count = 0
        if mode == "Prompt Template":
            instruction_count = st.number_input("Add Template", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add User Input", step=1, min_value=1, max_value=10)
            variable_count = st.number_input("Add Variable", step=1, min_value=1, max_value=10)
        elif model_type == "OpenAI Chat":
            instruction_count = st.number_input("Add System Message", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add User Message", step=1, min_value=1, max_value=10)
        else:
            instruction_count = st.number_input("Add Instruction", step=1, min_value=1, max_value=5)
            prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)
        var_names = []
        for i in range(variable_count):
            var_names.append(st.text_input(f"Variable {i+1} Name", value=f"Variable {i+1}", key=f"varname_{i}"))
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
                st.text_area("System Message" if model_type == "OpenAI Chat" else "Instruction", key=f"col_{j}")
            )

    prompts = []
    for i in range(prompt_count):
        cols = st.columns(instruction_count + 1)
        with cols[0]:
            prompts.append(st.text_area("User Message" if model_type == "OpenAI Chat" else "Prompt", key=f"row_{i}"))
        for j in range(1, instruction_count + 1):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    if st.button("Run"):
        df = load_data(model_type, model, instructions, prompts, temperature, api_key)
        for i in range(len(prompts)):
            for j in range(len(instructions)):
                placeholders[i][j + 1].markdown(df["response"][i + len(prompts) * j])
elif mode == "Prompt Template":
    instruction = None
    if model_type == "LlamaCpp Chat":
        instruction = st.text_area("Instruction", key="instruction")
    elif model_type == "OpenAI Chat":
        instruction = st.text_area("System Message", key="instruction")

    placeholders = [[st.empty() for _ in range(instruction_count + variable_count)] for _ in range(prompt_count)]

    cols = st.columns(instruction_count + variable_count)

    with cols[0]:
        a = None
    templates = []
    for j in range(variable_count, instruction_count + variable_count):
        with cols[j]:
            templates.append(st.text_area("Prompt Template", key=f"col_{j-variable_count}"))

    vars = []
    for i in range(prompt_count):
        cols = st.columns(instruction_count + variable_count)
        vars.append(dict())
        for j in range(variable_count):
            with cols[j]:
                vars[i][var_names[j]] = st.text_area(var_names[j], key=f"var_{i}_{j}")
        for j in range(variable_count, instruction_count + variable_count):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    if st.button("Run"):
        prompts = render_prompts(templates, vars)
        df = load_data(model_type, model, [instruction], prompts, temperature, api_key)
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
                instructions[j] = st.text_area("System Message", key=f"instruction_{j}")

    prompts = []
    for i in range(prompt_count):
        cols = st.columns(model_count + 1)
        with cols[0]:
            prompts.append(st.text_area("Prompt", key=f"row_{i}"))
        for j in range(1, model_count + 1):
            with cols[j]:
                placeholders[i][j] = st.empty()  # placeholders for the future output
        st.divider()

    if st.button("Run"):
        dfs = run_multiple(
            model_types, models, instructions, prompts, openai_api_key, anthropic_api_key, google_api_key, hf_api_key
        )
        for i in range(len(prompts)):
            for j in range(len(models)):
                placeholders[i][j + 1].markdown(dfs[j]["response"][i])
