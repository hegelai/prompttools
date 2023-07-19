Testing and CI/CD
===========

.. currentmodule:: prompttools.prompttest

After identifying the right evaluation/validation function for the outputs, you
can easily create unit tests and add them to your CI/CD workflow.

Unit tests in ``prompttools`` are called ``prompttests``. They use the ``@prompttest`` annotation to transform an
evaluation function into an efficient unit test. The ``prompttest`` framework executes and evaluates experiments
so you can test prompts over time. For example:

.. code-block:: python

    import prompttools.prompttest as prompttest

    @prompttest.prompttest(
        model_name="text-davinci-003",
        metric_name="similar_to_expected",
        prompt_template="Answer the following question: {{input}}",
        user_input=[{"input": "Who was the first president of the USA?"}],
    )
    def measure_similarity(
        input_pair: Tuple[str, Dict[str, str]], results: Dict, metadata: Dict
    ) -> float:
        r"""
        A simple test that checks semantic similarity between the user input
        and the model's text responses.
        """
        expected = EXPECTED[input_pair[1]["input"]]
        distances = [
            similarity.compute(expected, response)
            for response in extract_responses(results)
        ]
        return min(distances)


In the file, be sure to call the ``main()`` method of ``prompttest`` like you would for ``unittest``.

.. code-block:: python

    if __name__ == "__main__":
        prompttest.main()
