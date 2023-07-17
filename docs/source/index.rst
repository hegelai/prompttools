.. prompttools documentation master file, created by
   sphinx-quickstart on Sun Jul 16 15:34:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PromptTools!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Welcome to `prompttools` created by [Hegel AI](https://hegel-ai.com/)! 
This repo offers a set of free, open-source tools for testing and experimenting with prompts. 
The core idea is to enable developers to evaluate prompts using familiar interfaces 
like _code_ and _notebooks_.


To stay in touch with us about issues and future updates, 
join the [Discord](https://discord.gg/7KeRPNHGdJ).


Quickstart
==================

To install `prompttools`, you can use `pip`:

```
pip install prompttools
```

You can run a simple example of a `prompttools` with the following

```
DEBUG=1 python examples/prompttests/example.py
```

To run the example outside of `DEBUG` mode, you'll need to bring your own OpenAI API key. 
This is because `prompttools` makes a call to OpenAI from your machine. For example:

```
OPENAI_API_KEY=sk-... python examples/prompttests/example.py
```

You can see the full example [here](https://github.com/hegelai/prompttools/tree/main/examples/prompttests/test_openai_chat.py).

Using `prompttools`
==================

There are primarily two ways you can use `prompttools` in your LLM workflow:

1. Run experiments in [notebooks](https://github.com/hegelai/prompttools/tree/main/examples/notebooks/).
1. Write [unit tests](https://github.com/hegelai/prompttools/tree/main/examples/prompttests/test_openai_chat.py) and integrate them into your CI/CD workflow [via Github Actions](/.github/workflows/post-commit.yaml).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`