from typing import Callable, Dict, List


class ExperimentationHarness:
    """
    Base class for experimentation harnesses.
    """

    @staticmethod
    def _prepare_arguments(arguments: Dict[str, object]) -> Dict[str, List[object]]:
        return {name: [arg] for name, arg in arguments.items()}

    def prepare(self) -> None:
        """
        Prepares the underlying experiment.
        """
        self.experiment.prepare()

    def run(self) -> None:
        """
        Runs the underlying experiment.
        """
        self.experiment.run(self.PIVOT_COLUMNS[0], self.input_pairs_dict)

    def evaluate(
        self, metric_name: str, eval_fn: Callable, use_input_pairs: bool = False
    ) -> None:
        """
        Prepares the underlying experiment.
        """
        if use_input_pairs:
            self.experiment.evaluate(metric_name, eval_fn, self.input_pairs_dict)
        else:
            self.experiment.evaluate(metric_name, eval_fn)

    def gather_feedback(self) -> None:
        self.experiment.gather_feedback(self.input_pairs_dict, self.PIVOT_COLUMNS)

    def visualize(self, pivot: bool = False) -> None:
        if pivot:
            self.experiment.visualize(self.input_pairs_dict, self.PIVOT_COLUMNS)
        else:
            self.experiment.visualize()

    def rank(self, metric_name: str, is_average: bool = False) -> Dict[str, float]:
        return self.experiment.rank(
            self.input_pairs_dict, self.PIVOT_COLUMNS, metric_name, is_average
        )
