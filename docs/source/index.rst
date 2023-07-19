.. prompttools documentation master file, created by
   sphinx-quickstart on Sun Jul 16 15:34:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PromptTools
===========

Welcome to ``prompttools`` created by `Hegel
AI <https://hegel-ai.com/>`__! This repo offers a set of free,
open-source tools for testing and experimenting with prompts. The core
idea is to enable developers to evaluate prompts using familiar
interfaces like *code* and *notebooks*.

To stay in touch with us about issues and future updates, join the
`Discord <https://discord.gg/7KeRPNHGdJ>`__.

Installation
------------

To install ``prompttools`` using pip:

.. code:: bash

   pip install prompttools

To install from source, first clone this GitHub repo to your local
machine, then, from the repo, run:

.. code:: bash

   pip install .

You can then proceed to run `our examples <https://github.com/hegelai/prompttools/tree/main/examples/notebooks/>`__.

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
community. The current license can be found in the `LICENSE <https://github.com/hegelai/prompttools/tree/main/LICENSE>`__
file. If there is any concern, please `contact
us <mailto:eam@hegel-ai.com>`__ and we will be happy to work with you.

Module Index
-------

* :ref:`modindex`

.. Hidden TOCs

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   quickstart
   usage

.. toctree::
   :caption: Concepts
   :maxdepth: 2
   :hidden:

   experiment
   harness