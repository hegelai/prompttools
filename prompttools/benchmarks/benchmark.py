# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Optional
import pandas as pd
import warnings


class Benchmark:
    r"""
    Benchmark models using defined data sets.
    Find example under benchmarks/examples/benchmarking.ipynb.

    Args:
    ----
        experiment (experiment type): experiment to use
        eval_methods (list(eval methods)): list of evaluation methods to measure response similarity
        prompts (list(str)): list of queries, questions, prompts for LLMs to respond to
        response_options (list(str)): possible responses to measure against
        correct_response_index (list(int)): list of index of correct response in response_options
    """

    def __init__(
        self,
        experiment: Any,
        eval_method: Callable,
        prompts: List[str],
        response_options: List[Any],
        correct_response_indices: Optional[List[int]] = None,
    ):
        self.experiment = experiment
        self.eval_method = eval_method
        self.prompts = prompts
        self.response_options = response_options
        self.correct_response_indices = correct_response_indices

    def _get_precision(
        self,
        dataframe: pd.DataFrame,
        pred_col: str,
        label_col: str,
    ) -> float:
        r"""
        Calculate precision.
        """
        # TODO: coming soon
        pass

    def multiple_choice_accuracy(
        self,
        dataframe: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> float:
        r"""
        Benchmark LLM accuracy on multiple choice
        prompt endings.
        """
        correct = 0
        for _, row in dataframe.iterrows():
            if row[col1] == row[col2]:
                correct += 1
        return correct / len(dataframe)

    def multiple_choice_benchmark(
        self,
    ) -> Any:
        r"""
        Run model experiments to measure response quality.
        """
        self.experiment.run()

        if "prompt" not in self.experiment.full_df.columns:
            # Assume messages column is in place of prompt
            self.experiment.full_df["prompt"] = self.experiment.full_df["messages"].map(lambda x: str(x))
            warnings.warn("Column 'prompt' does not exist. Using column 'messages' instead.", UserWarning, stacklevel=2)
        # Get option with highest similarity to LLM response
        self.benchmark_df = self.experiment.full_df[["prompt", "response"]]
        self.benchmark_df["response_options"] = self.response_options
        self.benchmark_df = self.benchmark_df.explode(column="response_options").reset_index()
        scores = []
        for _, row in self.benchmark_df.iterrows():
            scores.append(self.eval_method(row=row, expected=row["response_options"]))
        self.benchmark_df["scores"] = scores
        self.benchmark_df['max_value'] = self.benchmark_df.groupby("prompt")["scores"].transform('max')
        self.benchmark_df = self.benchmark_df[self.benchmark_df["scores"] == self.benchmark_df["max_value"]]
        self.benchmark_df = self.benchmark_df.sort_index()
        # Colect model choices
        model_choice = []
        for i, choice in enumerate(self.benchmark_df["response_options"].values):
            model_choice.append(self.response_options[i].index(choice))
        self.benchmark_df["model_choice"] = model_choice
        self.benchmark_df["labels"] = self.correct_response_indices
        return self.multiple_choice_accuracy(self.benchmark_df, "model_choice", "labels")
