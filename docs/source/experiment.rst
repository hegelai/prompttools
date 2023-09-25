Experiment
==========================

.. currentmodule:: prompttools.experiment

There are two main abstractions used in the ``prompttools`` library: Experiments and Harnesses.
Occasionally, you may want to use a harness, because it abstracts away more details.

An experiment is a low level abstraction that takes the Cartesian product of possible inputs to
an LLM API. For example, the ``OpenAIChatExperiment`` accepts lists of inputs for each parameter
of the OpenAI Chat Completion API. Then, it constructs and asynchronously executes requests
using those potential inputs. An example of using experiment is `here <https://github.com/hegelai/prompttools/blob/main/examples/notebooks/BasicExperiment.ipynb>`_.

There are two ways to initialize an experiment:

  1. Wrap your parameters in ``list``\ s and pass them into the ``__init__`` method. See each class's
     method signature in the "Integrated Experiment APIs" section for details.
  2. Define which parameters should be tested and which ones should be frozen in two dictionaries. Pass the
     dictionaries to the ``initialize`` method. See the ``classmethod initialize`` below for details.

The ``Experiment`` superclass's shared API is below.

.. autoclass:: Experiment
    :members:

Integrated Experiment APIs
-----------------------------

LLMs
+++++++++++++++++++++++++++++++++++++++++

.. autoclass:: OpenAIChatExperiment

.. autoclass:: OpenAICompletionExperiment

.. autoclass:: AnthropicCompletionExperiment

.. autoclass:: HuggingFaceHubExperiment

.. autoclass:: GooglePaLMCompletionExperiment

.. autoclass:: GoogleVertexChatCompletionExperiment

.. autoclass:: LlamaCppExperiment

.. autoclass:: ReplicateExperiment

Frameworks
+++++++++++++++++++++++++++++++++++++++++

.. autoclass:: SequentialChainExperiment

.. autoclass:: RouterChainExperiment

.. autoclass:: MindsDBExperiment

Vector DBs
+++++++++++++++++++++++++++++++++++++++++

.. autoclass:: ChromaDBExperiment

.. autoclass:: WeaviateExperiment

.. autoclass:: LanceDBExperiment

.. autoclass:: QdrantExperiment

.. autoclass:: PineconeExperiment

Computer Vision
+++++++++++++++++++++++++++++++++++++++++

.. autoclass:: StableDiffusionExperiment

.. autoclass:: ReplicateExperiment
