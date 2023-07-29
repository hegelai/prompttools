import os
import streamlit as st
import jinja2

from prompttools.experiment import LlamaCppExperiment
from prompttools.selector.prompt_selector import PromptSelector

from prompttools.experiment import OpenAIChatExperiment
from prompttools.experiment import OpenAICompletionExperiment
from prompttools.experiment import LlamaCppExperiment
from prompttools.experiment import AnthropicCompletionExperiment
from prompttools.experiment import GooglePaLMCompletionExperiment
from prompttools.experiment import HuggingFaceHubExperiment

ENVIRONMENT_VARIABLE = {
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
}

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
    elif model_type == "OpenAI Chat":
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    return experiment.to_pandas_df()


with st.sidebar:
    mode = st.radio(
        "Choose a mode",
        ('Instruction', 'Prompt Template'))

    model_type = st.selectbox(
        "Model Type",
        ("OpenAI Chat", "OpenAI Completion", "Anthropic", "Google PaLM", "LlamaCpp Chat", "HuggingFace Hub"),
    )
    model, api_key = None, None
    if model_type == "LlamaCpp Chat":
        model = st.text_input("Local Model Path", key=f"llama_cpp_model_path")
    elif model_type == "HuggingFace Hub":
        model = st.text_input("Repo ID", key=f"hf_model_id")
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
    if mode == 'Prompt Template':
        instruction_count = st.number_input(
            "Add Template", step=1, min_value=1, max_value=5
        )
        prompt_count = st.number_input(
            "Add User Input", step=1, min_value=1, max_value=10
        )
        variable_count = st.number_input(
            "Add Variable", step=1, min_value=1, max_value=10
        )
    elif model_type == "OpenAI Chat":
        instruction_count = st.number_input(
            "Add System Prompt", step=1, min_value=1, max_value=5
        )
        prompt_count = st.number_input(
            "Add User Message", step=1, min_value=1, max_value=10
        )
    else:
        instruction_count = st.number_input(
            "Add Instruction", step=1, min_value=1, max_value=5
        )
        prompt_count = st.number_input(
            "Add Prompt", step=1, min_value=1, max_value=10
        )
    var_names = []
    for i in range(variable_count):
        var_names.append(st.text_input(f"Variable {i+1} Name", value=f"Variable {i+1}", key=f"varname_{i}"))

if mode == 'Instruction':
    placeholders = [[st.empty() for _ in range(instruction_count + 1)] for _ in range(prompt_count)]

    cols = st.columns(instruction_count + 1)

    with cols[0]:
        a = None
    instructions = []
    for j in range(1, instruction_count + 1):
        with cols[j]:
            instructions.append(
                st.text_area("System Prompt" if model_type == "OpenAI Chat" else "Instruction", key=f"col_{j}")
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
elif mode == 'Prompt Template':
    placeholders = [[st.empty() for _ in range(instruction_count + variable_count)] for _ in range(prompt_count)]

    cols = st.columns(instruction_count + variable_count)

    with cols[0]:
        a = None
    templates = []
    for j in range(variable_count, instruction_count + variable_count):
        with cols[j]:
            templates.append(
                st.text_area("Prompt Template", key=f"col_{j-variable_count}")
            )

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
        df = load_data(model_type, model, "You are a helpful assistant.", prompts, temperature, api_key)
        for i in range(len(prompts)):
            for j in range(len(templates)):
                placeholders[i][j + 1].markdown(df["response"][i + len(prompts) * j])