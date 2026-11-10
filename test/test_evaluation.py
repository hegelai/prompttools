from unittest import TestCase
from unittest.mock import Mock

from prompttools.experiment import Experiment


class LocalExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.all_args = {"prompt": ["first", "second"]}
        self.completion_fn = Mock(side_effect=lambda prompt: {"text": prompt})

    @staticmethod
    def _extract_responses(output):
        return output["text"]


class TestEvaluation(TestCase):
    def setUp(self):
        self.experiment = LocalExperiment()
        self.addCleanup(self.experiment.queue.shutdown)
        self.experiment.run()

    def assert_scores(self, metric_name, expected):
        for table in (self.experiment.score_df, self.experiment.partial_df, self.experiment.full_df):
            self.assertEqual(table[metric_name].tolist(), expected)
            self.assertEqual(table.columns.tolist().count(metric_name), 1)

    def test_recompute_metric_with_changed_expected_values(self):
        def matches(row, expected):
            return row["response"] == expected

        self.experiment.evaluate("matches", matches, expected=["first", "second"])
        self.assert_scores("matches", [True, True])

        self.experiment.evaluate("matches", matches, expected=["other", "second"])

        self.assert_scores("matches", [False, True])
        self.assertEqual(self.experiment.completion_fn.call_count, 2)

    def test_recompute_metric_with_changed_function_and_static_arguments(self):
        self.experiment.evaluate("score", lambda row: 1)
        self.experiment.evaluate("other", lambda row: 7)

        self.experiment.evaluate("score", lambda row, value: value, static_eval_fn_kwargs={"value": 2})

        self.assert_scores("score", [2, 2])
        self.assert_scores("other", [7, 7])
        self.assertEqual(self.experiment.get_table()["response"].tolist(), ["first", "second"])

    def test_failed_reevaluation_preserves_previous_scores(self):
        self.experiment.evaluate("score", lambda row: 1)

        def fails_on_second_row(row):
            if row["response"] == "second":
                raise ValueError("evaluation failed")
            return 2

        with self.assertRaisesRegex(ValueError, "evaluation failed"):
            self.experiment.evaluate("score", fails_on_second_row)

        self.assert_scores("score", [1, 1])
