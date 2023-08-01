# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, List
from collections import defaultdict
from prompttools.experiment import Experiment
import pandas as pd


class MultiExperimentHarness:
    r"""
    This is designed to run experiments across multiple model providers. The underlying APIs for different models
    (e.g. LlamaCpp and OpenAI) are different, this provides a way to manage that complexity.
    This will run experiments for different providers, and combine the results into a single table.

    The notebook "examples/notebooks/GPT4vsLlama2.ipynb" provides a good example how this can used
    to test prompts across different models.

    Args:
        experiments (list[Experiment]): The list of experiments that you would like to execute (e.g.
            ``prompttools.experiment.OpenAICompletionExperiment``)
    """

    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments

    def prepare(self):
        for experiment in self.experiments:
            experiment.prepare()

    def run(self):
        for experiment in self.experiments:
            experiment.run()

    def evaluate(self, metric_name: str, eval_fn: Callable) -> None:
        for experiment in self.experiments:
            experiment.evaluate(metric_name, eval_fn)

    def gather_feedback(self) -> None:
        pass

    def _get_argument_combos(self):
        tmp = [combo for experiment in self.experiments for combo in experiment.argument_combos]
        return tmp

    def _get_prompts(self):
        tmp = [combo for experiment in self.experiments for combo in experiment._get_prompts()]
        return tmp

    def _get_results(self):
        tmp = [
            experiment._extract_responses(result) for experiment in self.experiments for result in experiment.results
        ]
        return tmp

    def _get_scores(self):
        scores = defaultdict(list)
        for experiment in self.experiments:
            for name, score in experiment.scores.items():
                scores[name].extend(score)
        return scores

    def _get_experiment_names(self):
        tmp = [name for experiment in self.experiments for name in experiment._get_model_names()]
        return tmp

    def visualize(self, colname: str = None) -> None:
        scores = self._get_scores()
        data = {
            "prompt": self._get_prompts(),
            "response(s)": self._get_results(),
            "latency": scores["latency"],
            "model": self._get_experiment_names(),
        }
        # Add scores for each eval fn, including feedback
        for metric_name, evals in scores.items():
            if metric_name != "comparison":
                data[metric_name] = evals
        df = pd.DataFrame(data)
        if colname:
            df = pd.pivot_table(
                df,
                values=colname,
                index=["prompt"],
                columns=["model"],
                aggfunc=lambda x: x.iloc[0],
            )
        return df

    def rank(self, metric_name: str, is_average: bool = False) -> Dict[str, float]:
        pass
