import streamlit as st

from prompttools.experiment import LlamaCppExperiment
from prompttools.selector.prompt_selector import PromptSelector


@st.cache_data
def load_data(instructions, user_inputs, temperature=0.0):
    model_paths = ["/Users/stevenkrawczyk/Downloads/llama-2-7b-chat.ggmlv3.q2_K.bin"]
    selectors = [PromptSelector(instruction, user_input) for instruction in instructions for user_input in user_inputs]
    call_params = dict(temperature=[temperature])
    experiment = LlamaCppExperiment(model_paths, selectors, call_params=call_params)
    return experiment.to_pandas_df()


with st.sidebar:
    model_type = st.selectbox(
        'Model Type',
        ('Llama 2 Chat', 'OpenAI Chat', 'OpenAI Completion', 'Anthropic', 'Google PaLM', 'HuggingFace Hub'))
    if model_type == 'Llama 2 Chat':
        path = st.text_input('Local Model Path', key=f"llama_cpp_model_path")
    if model_type == 'HuggingFace Hub':
        path = st.text_input('Repo ID', key=f"hf_model_id")
    if model_type == 'OpenAI Chat':
         model_type = st.selectbox(
            'Model',
            ('gpt-3.5-turbo', 
             'gpt-3.5-turbo-16k', 
             'gpt-3.5-turbo-0613',
             'gpt-3.5-turbo-16k-0613',
             'gpt-3.5-turbo-0301',
             'gpt-4', 
             'gpt-4-0613', 
             'gpt-4-32k', 
             'gpt-4-32k-0613',
             'gpt-4-0314',
             'gpt-4-32k-0314'))
    if model_type == 'OpenAI Completion':
         model_type = st.selectbox(
            'Model',
            ('text-davinci-003', 
             'text-davinci-002', 
             'code-davinci-002'))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="temperature")
    instruction_count = st.number_input("Add Instruction", step=1, min_value=1, max_value=10)
    prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)

placeholders = [[st.empty() for _ in range(instruction_count)] for _ in range(prompt_count)]

cols=st.columns(instruction_count+1)

with cols[0]:
    a = None
instructions = []
for j in range(1, instruction_count+1):
    with cols[j]:
        instructions.append(st.text_area('Instruction', key=f"col_{j}"))

prompts = []
for i in range(prompt_count):
    cols=st.columns(instruction_count+1)
    with cols[0]:
        prompts.append(st.text_area('Prompt', key=f"row_{i}"))
    for j in range(1, instruction_count):
        with cols[j]:
            placeholders[i][j] = st.empty()  # placeholders for the future output
    st.divider()

if st.button('Run'):
    df = load_data(instructions, prompts, temperature)
    for i in range(len(prompts)):
        for j in range(len(instructions)):
            placeholders[i][j+1].markdown(df['response'][i + len(prompts) * j])

