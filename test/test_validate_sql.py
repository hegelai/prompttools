from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from prompttools.utils import validate_sql_response
from prompttools.utils.validate_sql import evaluate, validate


class TestSQLValidation(TestCase):
    def test_select_query_returns_a_float_score(self):
        score = validate("SELECT name FROM users WHERE id = 1")
        self.assertIsInstance(score, float)
        self.assertEqual(score, 1.0)

    def test_invalid_and_unsupported_queries_return_zero(self):
        for query in ["", "not sql", "SELECT * FROM", "DELETE FROM users"]:
            with self.subTest(query=query):
                self.assertEqual(validate(query), 0.0)

    def test_experiment_evaluator_supports_default_and_custom_columns(self):
        row = pd.Series({"response": "SELECT name FROM users", "generated_sql": "not sql"})
        self.assertEqual(validate_sql_response(row), 1.0)
        self.assertEqual(validate_sql_response(row, response_column_name="generated_sql"), 0.0)

    def test_prompttest_evaluator_returns_the_validation_score(self):
        self.assertEqual(evaluate("write a query", "SELECT name FROM users", {}), 1.0)
        self.assertEqual(evaluate("write a query", "not sql", {}), 0.0)

    def test_missing_optional_dependency_has_installation_guidance(self):
        with patch.dict("sys.modules", {"sqlvalidator": None}):
            with self.assertRaisesRegex(ModuleNotFoundError, r"prompttools\[sql\]"):
                validate("SELECT name FROM users")
