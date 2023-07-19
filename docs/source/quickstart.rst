Quickstart
===========

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
`here <https://github.com/hegelai/prompttools/tree/main/examples/prompttests/test_openai_chat.py>`__.

