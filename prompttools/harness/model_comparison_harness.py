# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Callable

import pandas as pd

from .harness import ExperimentationHarness
from prompttools.experiment import OpenAIChatExperiment
from .utility import is_interactive
from IPython import display
from tabulate import tabulate
import logging
from copy import deepcopy


class ModelComparisonHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for comparing models.

    Args:
        model_names (List[str]): The names of the models that you would like to compare
        system_prompts (List[str]): A list of system messages, one for each model.
        model_arguments (List[Optional[Dict]]): A list of model arguments, one for each model.
        user_messages (List[str]) User messages that will be tested across models. Defaults to ``[]``.
        runs (int): Number of runs to execute. Defaults to ``1``.
    """

    _experiment_type = "Comparison"
    PIVOT_COLUMNS = ["model", "messages"]

    def __init__(
        self,
        model_names: List[str],
        system_prompts: List[str],
        user_messages: List[str],
        model_arguments: List[Optional[dict]] = [],
        runs: int = 1,
    ):
        self.model_names = model_names
        self.system_prompts = system_prompts
        self.model_arguments = deepcopy(model_arguments)
        self.user_messages = user_messages
        self.runs = runs
        self.experiments = []

        if len(model_names) != len(system_prompts):
            raise RuntimeError(
                "The number of models must match the number of system messages, because each"
                "system prompt correspond to one model."
            )
        if model_arguments != [] and len(model_arguments) != len(model_names):
            raise RuntimeError(
                "The number of models must match the number of model argument dictionaries,"
                "because each dictionary of arguments correspond to one model."
            )
        super().__init__()

    def prepare(self) -> None:
        """
        Initializes and prepares the experiment.
        """
        self.experiments = []
        for i, model in enumerate(self.model_names):
            system_prompt = self.system_prompts[i]
            model_args = {} if self.model_arguments == [] else self.model_arguments[i]
            messages = []
            for message in self.user_messages:
                messages.append(
                    [
                        self._create_system_prompt(system_prompt),
                        self._create_human_message(message),
                    ]
                )

            experiment = OpenAIChatExperiment(
                [model],
                messages,
                **self._prepare_arguments(model_args),
            )
            self.experiments.append(experiment)

    @staticmethod
    def _create_system_prompt(content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}

    @staticmethod
    def _create_human_message(content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    @property
    def full_df(self):
        return self._full_df

    @property
    def partial_df(self):
        return self._partial_df

    @property
    def score_df(self):
        return self._score_df

    def run(self, clear_previous_results: bool = False):
        if not self.experiments:
            self.prepare()
        for exp in self.experiments:
            exp.run(clear_previous_results=clear_previous_results)
        self._update_dfs()

    def evaluate(self, metric_name: str, eval_fn: Callable, static_eval_fn_kwargs: dict = {}, **eval_fn_kwargs) -> None:
        r"""
        Uses the given eval_fn to evaluate the results of the underlying experiment.
        """
        if not self.experiments:
            raise RuntimeError("Cannot evaluate experiment without running first.")
        for exp in self.experiments:
            exp.evaluate(metric_name, eval_fn, static_eval_fn_kwargs, **eval_fn_kwargs)
        self._update_dfs()

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
                    if table[col].nunique() == 1  # Note this is checking for uniqueness
                ]
            )
            for col in columns_to_hide:
                if col in table.columns:
                    table = table.drop(col, axis=1)
            return table

    def _update_dfs(self):
        self._full_df = pd.concat([exp.full_df for exp in self.experiments], axis=0, ignore_index=True)
        self._partial_df = pd.concat([exp.partial_df for exp in self.experiments], axis=0, ignore_index=True)
        self._score_df = pd.concat([exp.score_df for exp in self.experiments], axis=0, ignore_index=True)

    def visualize(self, get_all_cols: bool = False):
        table = self.get_table(get_all_cols)
        if is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))

    def _get_state(self):
        state_params = {
            "model_names": self.model_names,
            "system_prompts": self.system_prompts,
            "user_messages": self.user_messages,
            "model_arguments": self.model_arguments,
            "child_experiment_states": [exp._get_state() for exp in self.experiments] if self.experiments else [],
        }
        state = (
            state_params,
            self.full_df,
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

        model_names = state_params["model_names"]
        system_prompts = state_params["system_prompts"]
        user_messages = state_params["user_messages"]
        model_arguments = state_params["model_arguments"]
        child_experiment_states = state_params["child_experiment_states"]

        harness = cls(model_names, system_prompts, user_messages, model_arguments)
        harness.experiments = [
            OpenAIChatExperiment._load_state(state, None, None, OpenAIChatExperiment._experiment_type)
            for state in child_experiment_states
        ]
        harness._update_dfs()
        harness._experiment_id = experiment_id
        harness._revision_id = revision_id
        print("Loaded harness.")
        return harness
