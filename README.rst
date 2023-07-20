PromptTools
===========

Welcome to ``prompttools`` created by `Hegel
AI <https://hegel-ai.com/>`__! This repo offers a set of free,
open-source tools for testing and experimenting with prompts. The core
idea is to enable developers to evaluate prompts using familiar
interfaces like *code* and *notebooks*.

To stay in touch with us about issues and future updates, join the
`Discord <https://discord.gg/7KeRPNHGdJ>`__.

Quickstart
----------

To install ``prompttools``, you can use ``pip``:

::

   pip install prompttools

You can run a simple example of a ``prompttools`` with the following

::

   DEBUG=1 python examples/prompttests/test_openai_chat.py

To run the example outside of ``DEBUG`` mode, you’ll need to bring your
own OpenAI API key. This is because ``prompttools`` makes a call to
OpenAI from your machine. For example:

::

   OPENAI_API_KEY=sk-... python examples/prompttests/test_openai_chat.py

You can see the full example
`here </examples/prompttests/test_openai_chat.py>`__.

Using ``prompttools``
---------------------

There are primarily two ways you can use ``prompttools`` in your LLM
workflow:

1. Run experiments in `notebooks </examples/notebooks/>`__.
2. Write `unit tests </examples/prompttests/test_openai_chat.py>`__ and
   integrate them into your CI/CD workflow `via Github
   Actions </.github/workflows/post-commit.yaml>`__.

Notebooks
~~~~~~~~~

There are a few different ways to run an experiment in a notebook.

The simplest way is to define an experimentation harness and an
evaluation function:

.. code:: python

   from prompttools.harness import PromptTemplateExperimentationHarness


   def eval_fn(prompt: str, results: Dict, metadata: Dict) -> float:
       # Your logic here, or use a built-in one such as `prompttools.utils.similarity`.
       pass

   prompt_templates = [
       "Answer the following question: {{input}}",
       "Respond the following query: {{input}}"
   ]

   user_inputs = [
       {"input": "Who was the first president?"},
       {"input": "Who was the first president of India?"}
   ]

   harness = PromptTemplateExperimentationHarness("text-davinci-003",
                                                  prompt_templates,
                                                  user_inputs)


   harness.run()
   harness.evaluate("metric_name", eval_fn)
   harness.visualize()  # The results will be displayed as a table in your notebook

.. figure:: img/table.png
   :alt: image

   image

If you are interested to compare different models, the `ModelComparison
example </examples/notebooks/ModelComparison.ipynb>`__ may be of
interest.

For an example of built-in evaluation function, please see this example
of `semantic similarity
comparison </examples/notebooks/SemanticSimilarity.ipynb>`__ for
details.

You can also manually enter feedback to evaluate prompts, see
`HumanFeedback.ipynb </examples/notebooks/HumanFeedback.ipynb>`__.

.. figure:: img/feedback.png
   :alt: image

   image

..

   Note: Above we used an ``ExperimentationHarness``. Under the hood,
   that harness uses an ``Experiment`` to construct and make API calls
   to LLMs. The harness is responsible for managing higher level
   abstractions, like prompt templates or system prompts. To see how
   experiments work at a low level, `see this
   example </examples/notebooks/BasicExperiment.ipynb>`__.

Unit Tests
~~~~~~~~~~

Unit tests in ``prompttools`` are called ``prompttests``. They use the
``@prompttest`` annotation to transform an evaluation function into an
efficient unit test. The ``prompttest`` framework executes and evaluates
experiments so you can test prompts over time. You can see an example
test `here </examples/prompttests/test_openai_chat.py>`__ and an example
of that test being used as a Github Action
`here </.github/workflows/post-commit.yaml>`__.

Persisting Results
~~~~~~~~~~~~~~~~~~

To persist the results of your tests and experiments, you can export your ``Experiment`` with the methods ``to_csv``,
``to_json``, or ``to_lora_json``. We are happy to further discuss your use cases, pain points, and what export
options may be useful for you.

Installation
------------

To install ``prompttools`` using pip:

.. code:: bash

   pip install prompttools

To install from source, first clone this GitHub repo to your local
machine, then, from the repo, run:

.. code:: bash

   pip install .

You can then proceed to run `our examples </examples/notebooks/>`__.

Frequently Asked Questions (FAQs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Will this library forward my LLM calls to a server before sending it
   to OpenAI/Anthropic/etc?

   -  No, the source code will be executed on your machine. Any call to
      LLM APIs will be directly executed from your machine without any
      forwarding.

Contributing
------------

We welcome PRs and suggestions! Don’t hesitate to open a PR/issue or to
reach out to us `via email <mailto:team@hegel-ai.com>`__. Please have a
look at our `contribution guide <CONTRIBUTING.md>`__ and `“Help Wanted”
issues <https://github.com/hegelai/prompttools/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`__
to get started!

Usage and Feedback
------------------

We will be delighted to work with early adopters to shape our designs.
Please reach out to us `via email <mailto:team@hegel-ai.com>`__ if
you’re interested in using this tooling for your project or have any
feedback.

License
-------

We will be gradually releasing more components to the open-source
community. The current license can be found in the `LICENSE <LICENSE>`__
file. If there is any concern, please `contact
us <mailto:eam@hegel-ai.com>`__ and we will be happy to work with you.
