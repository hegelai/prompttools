# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Type
from .harness import ExperimentationHarness, Experiment
import pandas as pd
from .utility import is_interactive
from IPython import display
from tabulate import tabulate
import logging


class SystemPromptExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test various system prompts.

    Args:
        experiment (Type[Experiment]): The experiment that you would like to execute (e.g.
            ``prompttools.experiment.OpenAICompletionExperiment``)
        model_name (str): The name of the model.
        system_prompts (List[str]): A list of system prompts for the model
        human_messages (List[str]): A list of human (user) messages to pass into the model
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``. Note that the values are not lists.
    """

    _experiment_type = "Instruction"
    PIVOT_COLUMNS = ["system_prompt", "user_input"]

    def __init__(
        self,
        experiment: Type[Experiment],
        model_name: str,
        system_prompts: List[str],
        human_messages: List[str],
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.experiment_cls_constructor = experiment
        self.model_name = model_name
        self.system_prompts = system_prompts
        self.human_messages = human_messages
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    @staticmethod
    def _create_system_prompt(content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}

    @staticmethod
    def _create_human_message(content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    def prepare(self) -> None:
        r"""
        Creates messages to use for the experiment, and then initializes and prepares the experiment.
        """
        self.input_pairs_dict = {}
        messages_to_try = []
        for system_prompt in self.system_prompts:
            for message in self.human_messages:
                history = [
                    self._create_system_prompt(system_prompt),
                    self._create_human_message(message),
                ]
                messages_to_try.append(history)
                self.input_pairs_dict[str(history)] = (system_prompt, message)
        self.experiment = self.experiment_cls_constructor(
            [self.model_name],
            messages_to_try,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self, clear_previous_results: bool = False):
        if not self.experiment:
            self.prepare()
        super().run(clear_previous_results=clear_previous_results)

    def _get_state(self):
        state_params = {
            "experiment_cls_constructor": self.experiment_cls_constructor,
            "model_name": self.model_name,
            "system_prompts": self.system_prompts,
            "human_messages": self.human_messages,
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
        system_prompts = state_params["system_prompts"]
        human_messages = state_params["human_messages"]
        model_arguments = state_params["model_arguments"]
        child_experiment_state = state_params["child_experiment_state"]

        harness = cls(experiment_cls_constructor, model_name, system_prompts, human_messages, model_arguments)
        harness.experiment = experiment_cls_constructor._load_state(
            child_experiment_state, None, None, experiment_cls_constructor._experiment_type
        )
        harness._experiment_id = experiment_id
        harness._revision_id = revision_id
        print("Loaded harness.")
        return harness

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
