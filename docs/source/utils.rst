Evaluation and Validation
===========

.. currentmodule:: prompttools.utils

These built-in functions help you to evaluate the outputs of your experiments.
They can also be used with ``prompttest`` for be part of your CI/CD system.

.. autofunction:: prompttools.utils.autoeval_binary_scoring

.. autofunction:: prompttools.utils.autoeval_scoring

.. autofunction:: prompttools.utils.autoeval_with_documents

.. autofunction:: prompttools.utils.chunk_text

.. autofunction:: prompttools.utils.compute_similarity_against_model

.. autofunction:: prompttools.utils.apply_moderation

.. autofunction:: prompttools.utils.ranking_correlation

.. autofunction:: prompttools.utils.validate_json_response

.. autofunction:: prompttools.utils.validate_json.validate_keys

.. autofunction:: prompttools.utils.validate_python_response

SQL validation
--------------

Install the optional parser with ``pip install 'prompttools[sql]'``. Score an
experiment's SQL responses without sending them to a database::

    from prompttools.utils import validate_sql_response

    experiment.evaluate("valid_sql", validate_sql_response)

Use ``prompttools.utils.validate_sql.evaluate`` with ``prompttest``. Both
interfaces return ``1.0`` when sqlvalidator accepts the text and ``0.0`` when
it rejects it. The parser supports SELECT statements and a limited set of
schemaless checks. A passing score does not prove dialect compatibility,
schema correctness, safe permissions, or successful execution.

.. autofunction:: prompttools.utils.validate_sql_response

.. autofunction:: prompttools.utils.semantic_similarity

.. autofunction:: prompttools.utils.structural_similarity

.. autofunction:: prompttools.utils.similarity.compute
