Testing and CI/CD
===========

.. currentmodule:: prompttools.prompttest

After identifying the right evaluation/validation function for the outputs, you
can easily create unit tests and add them to your CI/CD workflow.

Unit tests in ``prompttools`` are called ``prompttests``. They use the ``@prompttest`` annotation to transform a
completion function into an efficient unit test. The ``prompttest`` framework executes and evaluates experiments
so you can test prompts over time. For example:

.. code-block:: python

    import prompttools.prompttest as prompttest

    @prompttest.prompttest(
        metric_name="is_valid_json",
        eval_fn=validate_json.evaluate,
        prompts=[create_json_prompt()],
    )
    def json_completion_fn(prompt: str):
        response = None
        if os.getenv("DEBUG", default=False):
            response = mock_openai_completion_fn(**{"prompt": prompt})
        else:
            response = openai.completions.create(prompt)
        return response.choices[0].text


In the file, be sure to call the ``main()`` method of ``prompttest`` like you would for ``unittest``.

.. code-block:: python

    if __name__ == "__main__":
        prompttest.main()
