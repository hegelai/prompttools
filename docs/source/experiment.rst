Experiment
==========================

.. currentmodule:: prompttools.experiment

There are two main abstractions used in the ``prompttools`` library: Experiments and Harnesses.
Most of the time, you'll want to use a harness, because it abstracts away more details.

An experiment is a low level abstraction that takes the cartesian product of possible inputs to
an LLM API. For example, the ``OpenAIChatExperiment`` accepts lists of inputs for each parameter
of the OpenAI Chat Completion API. Then, it constructs and asynchronously executes requests
using those potential inputs. An example of using experiment is `here <https://github.com/hegelai/prompttools/blob/main/examples/notebooks/BasicExperiment.ipynb>`_.

.. autoclass:: Experiment
    :members:

.. autoclass:: OpenAIChatExperiment

.. autoclass:: OpenAICompletionExperiment

.. autoclass:: HuggingFaceHubExperiment

.. autoclass:: LlamaCppExperiment

.. autoclass:: ChromaDBExperiment

.. autoclass:: WeaviateExperiment
