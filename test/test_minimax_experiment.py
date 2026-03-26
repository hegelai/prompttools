# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

"""
Unit and integration tests for MiniMaxChatExperiment.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from prompttools.experiment import MiniMaxChatExperiment
from prompttools.mock.mock import mock_minimax_chat_completion_fn, DotDict


class TestMiniMaxChatExperimentImport(unittest.TestCase):
    """Test that MiniMaxChatExperiment can be imported properly."""

    def test_import_from_experiment(self):
        from prompttools.experiment import MiniMaxChatExperiment
        self.assertIsNotNone(MiniMaxChatExperiment)

    def test_in_all_exports(self):
        from prompttools.experiment import __all__
        self.assertIn("MiniMaxChatExperiment", __all__)


class TestMiniMaxChatExperimentInit(unittest.TestCase):
    """Test MiniMaxChatExperiment initialization."""

    def test_default_init(self):
        """Test that the experiment initializes with default parameters."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.all_args["model"], ["MiniMax-M2.5"])
        self.assertEqual(experiment.all_args["temperature"], [1.0])
        self.assertEqual(experiment.all_args["top_p"], [1.0])

    def test_multiple_models(self):
        """Test initialization with multiple models."""
        models = ["MiniMax-M2.7", "MiniMax-M2.5"]
        experiment = MiniMaxChatExperiment(
            model=models,
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        self.assertEqual(experiment.all_args["model"], models)

    def test_temperature_clamping_zero(self):
        """Test that temperature 0.0 is clamped to 0.01."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[0.0],
        )
        self.assertEqual(experiment.all_args["temperature"], [0.01])

    def test_temperature_clamping_high(self):
        """Test that temperature > 1.0 is clamped to 1.0."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[2.0],
        )
        self.assertEqual(experiment.all_args["temperature"], [1.0])

    def test_temperature_valid(self):
        """Test that a valid temperature passes through."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[0.7],
        )
        self.assertEqual(experiment.all_args["temperature"], [0.7])

    def test_temperature_none(self):
        """Test that None temperature is kept as None."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[None],
        )
        self.assertEqual(experiment.all_args["temperature"], [None])

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.7"],
            messages=[[{"role": "user", "content": "Hello"}]],
            temperature=[0.5],
            top_p=[0.9],
            max_tokens=[1024],
            stop=[["END"]],
        )
        self.assertEqual(experiment.all_args["max_tokens"], [1024])
        self.assertEqual(experiment.all_args["stop"], [["END"]])
        self.assertEqual(experiment.all_args["top_p"], [0.9])


class TestMiniMaxChatExperimentMethods(unittest.TestCase):
    """Test MiniMaxChatExperiment methods."""

    def setUp(self):
        self.experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[
                [{"role": "user", "content": "Who was the first president?"}],
            ],
            temperature=[0.5],
        )

    def test_is_chat(self):
        """Test that _is_chat returns True."""
        self.assertTrue(MiniMaxChatExperiment._is_chat())

    def test_prepare_creates_argument_combos(self):
        """Test that prepare creates argument combinations."""
        self.experiment.prepare()
        self.assertEqual(len(self.experiment.argument_combos), 1)
        self.assertEqual(self.experiment.argument_combos[0]["model"], "MiniMax-M2.5")

    def test_prepare_cartesian_product(self):
        """Test that prepare creates cartesian product of arguments."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5", "MiniMax-M2.7"],
            messages=[
                [{"role": "user", "content": "Hello"}],
            ],
            temperature=[0.5, 1.0],
        )
        experiment.prepare()
        # 2 models * 1 message * 2 temperatures = 4 combos
        self.assertEqual(len(experiment.argument_combos), 4)

    def test_get_model_names(self):
        """Test _get_model_names returns correct model names."""
        self.experiment.prepare()
        names = self.experiment._get_model_names()
        self.assertEqual(names, ["MiniMax-M2.5"])

    def test_extract_responses(self):
        """Test _extract_responses with mock response."""
        response = mock_minimax_chat_completion_fn()
        result = MiniMaxChatExperiment._extract_responses(response)
        self.assertEqual(result, "George Washington")

    def test_url(self):
        """Test that the URL is set correctly."""
        self.assertEqual(
            MiniMaxChatExperiment.url,
            "https://api.minimax.io/v1/chat/completions",
        )


class TestMiniMaxChatMockFunction(unittest.TestCase):
    """Test the mock function for MiniMax chat completions."""

    def test_mock_returns_dotdict(self):
        """Test that mock returns a DotDict."""
        result = mock_minimax_chat_completion_fn()
        self.assertIsInstance(result, DotDict)

    def test_mock_has_choices(self):
        """Test that mock response has choices."""
        result = mock_minimax_chat_completion_fn()
        self.assertIn("choices", result)
        self.assertEqual(len(result["choices"]), 1)

    def test_mock_content(self):
        """Test mock response content."""
        result = mock_minimax_chat_completion_fn()
        self.assertEqual(result.choices[0].message.content, "George Washington")

    def test_mock_model(self):
        """Test mock response model."""
        result = mock_minimax_chat_completion_fn()
        self.assertEqual(result.model, "MiniMax-M2.5")

    def test_mock_usage(self):
        """Test mock response usage."""
        result = mock_minimax_chat_completion_fn()
        self.assertEqual(result.usage.total_tokens, 75)

    def test_mock_finish_reason(self):
        """Test mock response finish_reason."""
        result = mock_minimax_chat_completion_fn()
        self.assertEqual(result.choices[0].finish_reason, "stop")


class TestMiniMaxChatExperimentDebugMode(unittest.TestCase):
    """Test MiniMaxChatExperiment in DEBUG mode."""

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_uses_mock(self):
        """Test that DEBUG mode uses mock completion function."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        self.assertEqual(experiment.completion_fn, mock_minimax_chat_completion_fn)

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_run(self):
        """Test running the experiment in DEBUG mode."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Who was the first president?"}]],
        )
        experiment.run()
        self.assertIsNotNone(experiment.full_df)
        self.assertIsNotNone(experiment.partial_df)
        self.assertIsNotNone(experiment.score_df)

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_run_extracts_response(self):
        """Test that DEBUG run extracts response correctly."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Who was the first president?"}]],
        )
        experiment.run()
        self.assertIn("response", experiment.partial_df.columns)
        self.assertEqual(experiment.partial_df["response"].iloc[0], "George Washington")

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_run_multiple_models(self):
        """Test DEBUG run with multiple models."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5", "MiniMax-M2.7"],
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        experiment.run()
        self.assertEqual(len(experiment.full_df), 2)

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_run_has_latency(self):
        """Test that DEBUG run records latency."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        experiment.run()
        self.assertIn("latency", experiment.score_df.columns)
        self.assertGreaterEqual(experiment.score_df["latency"].iloc[0], 0)

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_run_to_csv(self):
        """Test exporting DEBUG run results to CSV."""
        import tempfile
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "Hello"}]],
        )
        experiment.run()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            experiment.to_csv(f.name)
            self.assertTrue(os.path.exists(f.name))

    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_cartesian_product(self):
        """Test cartesian product of arguments in DEBUG mode."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5", "MiniMax-M2.7"],
            messages=[
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "World"}],
            ],
            temperature=[0.5, 1.0],
        )
        experiment.run()
        # 2 models * 2 messages * 2 temperatures = 8
        self.assertEqual(len(experiment.full_df), 8)


class TestMiniMaxChatExperimentPlayground(unittest.TestCase):
    """Test MiniMax integration in playground constants."""

    def test_playground_experiment_registered(self):
        from prompttools.playground.constants import EXPERIMENTS
        self.assertIn("MiniMax Chat", EXPERIMENTS)
        self.assertEqual(EXPERIMENTS["MiniMax Chat"], MiniMaxChatExperiment)

    def test_playground_env_var_registered(self):
        from prompttools.playground.constants import ENVIRONMENT_VARIABLE
        self.assertIn("MiniMax Chat", ENVIRONMENT_VARIABLE)
        self.assertEqual(ENVIRONMENT_VARIABLE["MiniMax Chat"], "MINIMAX_API_KEY")

    def test_playground_model_type_registered(self):
        from prompttools.playground.constants import MODEL_TYPES
        self.assertIn("MiniMax Chat", MODEL_TYPES)

    def test_playground_minimax_models(self):
        from prompttools.playground.constants import MINIMAX_CHAT_MODELS
        self.assertIn("MiniMax-M2.7", MINIMAX_CHAT_MODELS)
        self.assertIn("MiniMax-M2.5", MINIMAX_CHAT_MODELS)


class TestMiniMaxChatExperimentInitialize(unittest.TestCase):
    """Test the alternative initialize() class method."""

    def test_initialize_class_method(self):
        """Test initialization via the initialize class method."""
        test_params = {"model": ["MiniMax-M2.5", "MiniMax-M2.7"]}
        frozen_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
        }
        experiment = MiniMaxChatExperiment.initialize(test_params, frozen_params)
        self.assertIsNotNone(experiment)
        self.assertEqual(len(experiment.all_args["model"]), 2)


class TestMiniMaxChatIntegration(unittest.TestCase):
    """Integration tests for MiniMax chat experiment.

    These tests require a valid MINIMAX_API_KEY environment variable.
    They are skipped if the key is not available.
    """

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_live_single_completion(self):
        """Test a live single completion call to MiniMax API."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "What is 2+2? Answer with just the number."}]],
            temperature=[0.01],
            max_tokens=[32],
        )
        experiment.run()
        self.assertIsNotNone(experiment.full_df)
        response = experiment.partial_df["response"].iloc[0]
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_live_model_comparison(self):
        """Test comparing two MiniMax models."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5", "MiniMax-M2.5-highspeed"],
            messages=[[{"role": "user", "content": "Say hello in one word."}]],
            temperature=[0.5],
            max_tokens=[32],
        )
        experiment.run()
        self.assertEqual(len(experiment.full_df), 2)

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_live_evaluate(self):
        """Test running evaluation on live results."""
        experiment = MiniMaxChatExperiment(
            model=["MiniMax-M2.5"],
            messages=[[{"role": "user", "content": "What is the capital of France?"}]],
            temperature=[0.5],
            max_tokens=[64],
        )
        experiment.run()

        def contains_paris(row, response_column_name="response"):
            return "paris" in row[response_column_name].lower()

        experiment.evaluate("contains_paris", contains_paris)
        self.assertIn("contains_paris", experiment.score_df.columns)


if __name__ == "__main__":
    unittest.main()
