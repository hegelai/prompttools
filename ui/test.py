import streamlit as st
import pandas as pd
import numpy as np

from prompttools.experiment import LlamaCppExperiment
from prompttools.selector.prompt_selector import PromptSelector


# for i in range(len(data['tasks'])):
#     title = st.title(data['tasks'][i]['name'])
#     form = st.form(key=f"form_one_{i}")
#     input = form.text_input(label=f"{data['tasks'][i]['inputs']}", key=f"input1_{i}")
#     input_two = form.text_input(label=f"{data['tasks'][i]['inputs_two']}", key=f"input2_{i}")
#     form.form_submit_button()

instruction = st.text_input(label="Instruction", value="Answer the following question")
user_input_count = st.number_input("Add User Inputs", step=1, min_value=1, max_value=10)
user_inputs = [st.text_input(label="User Input", key=f"user_input_{i}") for i in range(user_input_count)]
temperature_count = st.number_input("Add Temperatures", step=1, min_value=1, max_value=10)
temperatures = [
    st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"temperature_{i}")
    for i in range(temperature_count)
]
# st.button("Add User Input", on_click=state.increment_user_input)

# st.button("Add Temperature", on_click=state.increment_user_input)
submit_button = st.button(label="Submit")


@st.cache_data
def load_data(instruction, user_inputs, temperatures):
    model_paths = ["/Users/stevenkrawczyk/Downloads/llama-2-7b-chat.ggmlv3.q2_K.bin"]
    selectors = [PromptSelector(instruction, user_input) for user_input in user_inputs]
    call_params = dict(temperature=temperatures)
    experiment = LlamaCppExperiment(model_paths, selectors, call_params=call_params)
    return experiment.to_pandas_df()


if submit_button:
    data = load_data(instruction, user_inputs, temperatures)
    st.table(data)
