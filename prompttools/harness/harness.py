# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
from prompttools.experiment import Experiment


from prompttools.common import HEGEL_BACKEND_URL
import os
import pickle
import requests


class ExperimentationHarness:
    r"""
    Base class for experimentation harnesses. This should not be used directly, please use the subclasses instead.
    """

    experiment: Experiment
    PIVOT_COLUMNS: list

    def __init__(self) -> None:
        self.input_pairs_dict = None
        self.experiment = None
        self.runs = 1
        self._experiment_id = None
        self._revision_id = None

    @staticmethod
    def _prepare_arguments(arguments: dict[str, object]) -> dict[str, list[object]]:
        return {name: [arg] for name, arg in arguments.items()}

    def prepare(self) -> None:
        r"""
        Prepares the underlying experiment.
        """
        self.experiment.prepare()

    def run(self, clear_previous_results: bool = False) -> None:
        r"""
        Runs the underlying experiment.
        """
        self.experiment.run(runs=self.runs, clear_previous_results=clear_previous_results)

    def evaluate(self, metric_name: str, eval_fn: Callable, static_eval_fn_kwargs: dict = {}, **eval_fn_kwargs) -> None:
        r"""
        Uses the given eval_fn to evaluate the results of the underlying experiment.
        """
        self.experiment.evaluate(metric_name, eval_fn, static_eval_fn_kwargs, **eval_fn_kwargs)

    # def gather_feedback(self) -> None:
    #     self.experiment.gather_feedback(self.input_pairs_dict, self.PIVOT_COLUMNS)

    def visualize(self, pivot: bool = False) -> None:
        r"""
        Displays a visualization of the experiment results.
        """
        if pivot:
            self.experiment.visualize(pivot_columns=self.PIVOT_COLUMNS, pivot=True)
        else:
            self.experiment.visualize()

    def rank(self, metric_name: str, is_average: bool = False) -> dict[str, float]:
        r"""
        Scores and ranks the experiment inputs using the pivot columns,
        e.g. prompt templates or system prompts.
        """
        return self.experiment.rank(self.input_pairs_dict, self.PIVOT_COLUMNS, metric_name, is_average)

    @property
    def full_df(self):
        return self.experiment.full_df

    @property
    def partial_df(self):
        return self.experiment.partial_df

    @property
    def score_df(self):
        return self.experiment.score_df

    def _get_state(self):
        raise NotImplementedError("Should be implemented by specific harness class.")

    @classmethod
    def _load_state(cls, state, experiment_id: str, revision_id: str, experiment_type_str: str):
        raise NotImplementedError("Should be implemented by specific harness class.")

    def save_experiment(self, name: Optional[str] = None):
        r"""
        name (str, optional): Name of the experiment. This is optional if you have previously loaded an experiment
            into this object.
        """
        if name is None and self._experiment_id is None:
            raise RuntimeError("Please provide a name for your experiment.")
        if self.full_df is None:
            raise RuntimeError("Cannot save empty experiment. Please run it first.")
        if os.environ["HEGELAI_API_KEY"] is None:
            raise PermissionError("Please set HEGELAI_API_KEY (e.g. os.environ['HEGELAI_API_KEY']).")
        state = self._get_state()
        url = f"{HEGEL_BACKEND_URL}/sdk/save"
        headers = {
            "Content-Type": "application/octet-stream",  # Use a binary content type for pickled data
            "Authorization": os.environ["HEGELAI_API_KEY"],
        }
        print("Sending HTTP POST request...")
        data = pickle.dumps((name, self._experiment_id, self._experiment_type, state))
        response = requests.post(url, data=data, headers=headers)
        self._experiment_id = response.json().get("experiment_id")
        self._revision_id = response.json().get("revision_id")
        return response

    @classmethod
    def load_experiment(cls, experiment_id: str):
        r"""
        experiment_id (str): experiment ID of the experiment that you wish to load.
        """
        if os.environ["HEGELAI_API_KEY"] is None:
            raise PermissionError("Please set HEGELAI_API_KEY (e.g. os.environ['HEGELAI_API_KEY']).")

        url = f"{HEGEL_BACKEND_URL}/sdk/get/experiment/{experiment_id}"
        headers = {
            "Content-Type": "application/octet-stream",  # Use a binary content type for pickled data
            "Authorization": os.environ["HEGELAI_API_KEY"],
        }
        print("Sending HTTP GET request...")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:  # Note that state should not have `name` included
            new_experiment_id, revision_id, experiment_type_str, state = pickle.loads(response.content)
            if new_experiment_id != experiment_id:
                raise RuntimeError("Experiment ID mismatch between request and response.")
            return cls._load_state(state, experiment_id, revision_id, experiment_type_str)
        else:
            print(f"Error: {response.status_code}, {response.text}")

    @classmethod
    def load_revision(cls, revision_id: str):
        r"""
        revision_id (str): revision ID of the experiment that you wish to load.
        """
        if os.environ["HEGELAI_API_KEY"] is None:
            raise PermissionError("Please set HEGELAI_API_KEY (e.g. os.environ['HEGELAI_API_KEY']).")

        url = f"{HEGEL_BACKEND_URL}/sdk/get/revision/{revision_id}"
        headers = {
            "Content-Type": "application/octet-stream",  # Use a binary content type for pickled data
            "Authorization": os.environ["HEGELAI_API_KEY"],
        }
        print("Sending HTTP GET request...")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            experiment_id, new_revision_id, experiment_type_str, state = pickle.loads(response.content)
            if new_revision_id != revision_id:
                raise RuntimeError("Revision ID mismatch between request and response.")
            return cls._load_state(state, experiment_id, revision_id, experiment_type_str)
        else:
            print(f"Error: {response.status_code}, {response.text}")
