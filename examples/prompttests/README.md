# Prompt tests

## Citation preservation with captured search evidence

`test_citation_preservation.py` compares two prompt variants on the same public
Python documentation evidence using the existing `@prompttest` completion and
evaluation workflow. It prints each answer and a pass/fail result, and exits with
status 1 if either variant fails. This makes a removed URL or supporting quote
visible when changing a prompt.

Install PromptTools from the repository root:

```sh
pip install -e .
```

Set `OPENAI_API_KEY` and `PROMPTTOOLS_MODEL` in your shell for your chat completion
provider, then run from the repository root:

```sh
SENTRY_OPT_OUT=1 python examples/prompttests/test_citation_preservation.py --model "$PROMPTTOOLS_MODEL"
```

The default uses a small captured search transcript in
`fixtures/parallel_search.json`. Both variants receive exactly the same evidence,
so a changing search result does not confound the prompt comparison. Generation
still calls your configured provider and may incur its usual costs. The example
uses the OpenAI client, including its `OPENAI_BASE_URL` configuration for compatible
endpoints. It does not require a Parallel key.

To replace the snapshot with one live, anonymous Parallel MCP search before
running both variants:

```sh
pip install 'mcp==1.26.0'
SENTRY_OPT_OUT=1 python examples/prompttests/test_citation_preservation.py --model "$PROMPTTOOLS_MODEL" --live-search
```

See [Parallel MCP setup](https://docs.parallel.ai/integrations/mcp/quickstart).
Anonymous search is rate limited. Search failures and provider errors propagate;
the example does not silently replace failed live searches with captured results.
The optional search sends only the fixed public documentation query. The selected
search evidence and question are then sent to your configured completion provider.

The response format is JSON with an `answer` and a nonempty `citations` list. Each
citation must preserve a returned source URL exactly and quote at least 20
characters verbatim from that URL's excerpts. Every supplied citation is checked.
Here, “canonical” means the URL returned in the evidence; URLs are not rewritten.
The score checks URL and quotation preservation. It does not establish that the
quote supports the answer's claims, or that the answer covers every source.
Inspect the printed answers for those properties. A single question is a runnable
regression example, not a prompt quality benchmark.

The fixture retains one URL and one sentence from an anonymous `web_search` call
on 2026-11-17; other results and excerpt text were omitted. It contains no generated
answers. The question and search arguments are in the example so it can be rerun.
Source: [Python asyncio runners documentation](https://docs.python.org/3/library/asyncio-runner.html).

This example was contributed by George Pickett, who works at Parallel, with AI
assistance.
