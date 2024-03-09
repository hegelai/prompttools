Harness
===========

.. currentmodule:: prompttools.harness

There are two main abstractions used in the ``prompttools`` library: Experiments and Harnesses.
Occasionally, you may want to use a harness, because it abstracts away more details.

A harness is built on top of an experiment, and manages abstractions over inputs.
For example, the ``PromptTemplateExperimentationHarness`` freezes one set of model arguments
and varies the prompt input based on prompt templates and user inputs. It then constructs
a corresponding experiment, and keeps track of the templates and inputs used for each prompt.

.. autoclass:: ExperimentationHarness
    :members:

.. autoclass:: ChatHistoryExperimentationHarness

.. autoclass:: ChatModelComparisonHarness

.. autoclass:: ChatPromptTemplateExperimentationHarness

.. autoclass:: ModelComparisonHarness

.. autoclass:: MultiExperimentHarness

.. autoclass:: PromptTemplateExperimentationHarness

.. autoclass::  RetrievalAugmentedGenerationExperimentationHarness

.. autoclass:: SystemPromptExperimentationHarness
