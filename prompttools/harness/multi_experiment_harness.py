from typing import Callable, Dict, List
from collections import defaultdict
from prompttools.experiment import Experiment
import pandas as pd


class MultiExperimentHarness:
    def __init__(self, experiments: List[Experiment], prompts: List[str] = []):
        self.experiments = experiments
        self.prompts = prompts

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
        argument_combos = self._get_argument_combos()
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
