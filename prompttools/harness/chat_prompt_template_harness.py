# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Type
import jinja2
import pandas as pd

from .harness import ExperimentationHarness, Experiment
from typing import Optional
from copy import deepcopy
from .utility import is_interactive
from IPython import display
from tabulate import tabulate
import logging


def _render_messages_openai_chat(message_template: list[dict], user_input: dict, environment):
    rendered_message = deepcopy(message_template)
    sys_msg_template = environment.from_string(rendered_message[0]["content"])
    user_msg_template = environment.from_string(rendered_message[-1]["content"])
    rendered_message[0]["content"] = sys_msg_template.render(**user_input)
    rendered_message[-1]["content"] = user_msg_template.render(**user_input)
    return rendered_message


class ChatPromptTemplateExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test various prompt templates for chat models.
    We use `jinja` templates, e.g. "Answer the following question: {{input}}".

    Args:
        experiment (Type[Experiment]): The experiment constructor that you would like to execute within the harness
            (e.g. ``prompttools.experiment.OpenAICompletionExperiment``)
        model_name (str): The name of the model.
        message_templates (List[str]): A list of prompt ``jinja``-styled templates. Each template should have two
            messages inside (first system prompt and second a user message).
        user_inputs (List[Dict[str, str]]): A list of dictionaries representing user inputs.
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``. Note that the values are not lists.
    """

    _experiment_type = "Template"
    PIVOT_COLUMNS = ["prompt_template", "user_input"]

    def __init__(
        self,
        experiment: Type[Experiment],
        model_name: str,
        message_templates: list[list[dict]],
        user_inputs: list[dict[str, str]],
        model_arguments: Optional[dict[str, object]] = None,
    ):
        self.environment = jinja2.Environment()
        self.experiment_cls_constructor = experiment
        self.model_name = model_name
        self.message_templates = message_templates
        self.user_inputs = user_inputs
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    def prepare(self) -> None:
        r"""
        Creates prompts from templates to use for the experiment, and then initializes and prepares the experiment.
        """
        # self.input_pairs_dict = {}
        rendered_messages = []
        for mt in self.message_templates:
            for user_input in self.user_inputs:
                rendered_messages.append(_render_messages_openai_chat(mt, user_input, self.environment))
        self.experiment = self.experiment_cls_constructor(
            [self.model_name],
            rendered_messages,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self, clear_previous_results: bool = False):
        if not self.experiment:
            self.prepare()
        super().run(clear_previous_results=clear_previous_results)

        # Add user inputs to DataFrame
        if len(self.experiment.full_df) > 0:
            repeat = len(self.experiment.full_df) // len(self.user_inputs)
            user_inputs = deepcopy(self.user_inputs)
            user_inputs_col_name = "user_inputs"
            user_input_df = pd.DataFrame({user_inputs_col_name: user_inputs * repeat})

            if user_inputs_col_name in self.experiment.full_df.columns:
                self.experiment.full_df = self.experiment.full_df.drop(user_inputs_col_name, axis=1)
            self.experiment.full_df.reset_index(drop=True, inplace=True)

            self.experiment.full_df = pd.concat([user_input_df, self.experiment.full_df], axis=1)
            if user_inputs_col_name in self.experiment.partial_df.columns:
                self.experiment.partial_df = self.experiment.partial_df.drop(user_inputs_col_name, axis=1)
            self.experiment.partial_df.reset_index(drop=True, inplace=True)
            self.experiment.partial_df = pd.concat([user_input_df, self.experiment.partial_df], axis=1)

    def get_table(self, get_all_cols: bool = False) -> pd.DataFrame:
        columns_to_hide = [
            "stream",
            "response_id",
            "response_choices",
            "response_created",
            "response_created",
            "response_object",
            "response_model",
            "response_system_fingerprint",
            "revision_id",
            "log_id",
        ]

        if get_all_cols:
            return self.full_df
        else:
            table = self.full_df
            columns_to_hide.extend(
                [
                    col
                    for col in ["temperature", "top_p", "n", "presence_penalty", "frequency_penalty"]
                    if col not in self.partial_df.columns
                ]
            )
            for col in columns_to_hide:
                if col in table.columns:
                    table = table.drop(col, axis=1)
            return table

    def visualize(self, get_all_cols: bool = False):
        table = self.get_table(get_all_cols)
        if is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))

    def _get_state(self):
        state_params = {
            "experiment_cls_constructor": self.experiment_cls_constructor,
            "model_name": self.model_name,
            "message_templates": self.message_templates,
            "user_inputs": self.user_inputs,
            "model_arguments": self.model_arguments,
            "child_experiment_state": self.experiment._get_state() if self.experiment else None,
        }
        state = (
            state_params,
            self.experiment.full_df,
        )
        print("Creating state of experiment...")
        return state

    @classmethod
    def _load_state(cls, state, experiment_id: str, revision_id: str, experiment_type_str: str):
        (
            state_params,
            full_df,
        ) = state
        if experiment_type_str != cls._experiment_type:
            raise RuntimeError(
                f"The Experiment Type you are trying to load is {experiment_type_str},"
                "which does not match the current class."
            )

        experiment_cls_constructor = state_params["experiment_cls_constructor"]
        model_name = state_params["model_name"]
        message_templates = state_params["message_templates"]
        user_inputs = state_params["user_inputs"]
        model_arguments = state_params["model_arguments"]
        child_experiment_state = state_params["child_experiment_state"]

        harness = cls(experiment_cls_constructor, model_name, message_templates, user_inputs, model_arguments)
        harness.experiment = experiment_cls_constructor._load_state(
            child_experiment_state, None, None, experiment_cls_constructor._experiment_type
        )
        harness._experiment_id = experiment_id
        harness._revision_id = revision_id
        print("Loaded harness.")
        return harness
