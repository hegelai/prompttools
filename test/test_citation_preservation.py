import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples/prompttests/test_citation_preservation.py"
SPEC = importlib.util.spec_from_file_location("citation_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
SOURCES = example.captured_sources()
URL = SOURCES[0]["url"]
QUOTE = SOURCES[0]["excerpts"][0]


def answer(url=URL, quote=QUOTE):
    return {"answer": "It executes the coroutine and returns its result.", "citations": [{"url": url, "quote": quote}]}


@pytest.mark.parametrize(
    "response, expected_score",
    [
        (answer(), 1.0),
        (answer(url=URL + "?changed=1"), 0.0),
        (answer(url="https://example.com/?source=" + URL), 0.0),
        (answer(quote="A quotation that does not occur in the evidence."), 0.0),
        (answer(quote="Execute"), 0.0),
        (answer(url=[URL]), 0.0),
        ({"answer": "No citations", "citations": []}, 0.0),
        ({"answer": " ", "citations": answer()["citations"]}, 0.0),
        ({"answer": "Text", "citations": [answer()["citations"][0], {"url": URL, "quote": "invented"}]}, 0.0),
        ({"answer": "Text", "citations": [None]}, 0.0),
        ([], 0.0),
        (None, 0.0),
    ],
)
def test_citation_score(response, expected_score):
    assert example.citation_score("unused", json.dumps(response), {}, json.dumps(SOURCES)) == expected_score


def test_rejects_invalid_json():
    assert example.citation_score("unused", "not JSON", {}, json.dumps(SOURCES)) == 0.0


def test_quotes_must_belong_to_the_cited_source():
    sources = SOURCES + [
        {"url": "https://example.com/other", "excerpts": ["Completely different supporting evidence."]}
    ]
    response = json.dumps(answer(url="https://example.com/other"))
    assert example.citation_score("unused", response, {}, json.dumps(sources)) == 0.0


@pytest.mark.parametrize("scenario", ["success", "tool_error", "empty"])
def test_optional_mcp_search(scenario):
    pytest.importorskip("mcp")
    import httpx

    calls = []
    real_client = httpx.AsyncClient

    def handle(request):
        assert request.headers["User-Agent"] == example.USER_AGENT
        assert "Authorization" not in request.headers
        assert "x-api-key" not in request.headers
        message = json.loads(request.content)
        calls.append(message)
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert method == "tools/call"
            assert message["params"]["name"] == "web_search"
            assert message["params"]["arguments"] == example.SEARCH_ARGUMENTS
            result = {
                "content": [{"type": "text", "text": "search error" if scenario == "tool_error" else "results"}],
                "isError": scenario == "tool_error",
                "structuredContent": {"results": SOURCES if scenario == "success" else []},
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def client_factory(**kwargs):
        assert kwargs["timeout"].connect == 30
        assert kwargs["timeout"].read == 60
        return real_client(**kwargs, transport=httpx.MockTransport(handle))

    with patch("httpx.AsyncClient", client_factory):
        if scenario == "success":
            assert asyncio.run(example.search_sources()) == SOURCES
        else:
            with pytest.raises(Exception) as error:
                asyncio.run(example.search_sources())

            # Context managers may wrap the original RuntimeError in an ExceptionGroup.
            def messages(exception):
                return str(exception) + " ".join(messages(child) for child in getattr(exception, "exceptions", []))

            expected = "Search failed" if scenario == "tool_error" else "no Python documentation excerpts"
            assert expected in messages(error.value)
    assert any(call["method"] == "tools/call" for call in calls)


@pytest.mark.parametrize("fail_concise", [False, True])
def test_cli_runs_both_prompt_variants(fail_concise):
    requests = []

    class CompletionHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert self.headers["Authorization"] == "Bearer example-test-value"
            assert self.path == "/v1/chat/completions"
            prompt = body["messages"][0]["content"]
            assert json.loads(prompt.split("Sources: ", 1)[1]) == SOURCES
            content = answer()
            if fail_concise and prompt.startswith(example.PROMPT_VARIANTS["concise"]):
                content["citations"] = []
            payload = json.dumps(
                {"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE_PATH), "--model", "test-deployment"],
            cwd=ROOT,
            env={
                "PATH": os.defpath,
                "PYTHONPATH": str(ROOT),
                "SENTRY_OPT_OUT": "1",
                "OPENAI_API_KEY": "example-test-value",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result.returncode == int(fail_concise), result.stdout + result.stderr
    assert len(requests) == 2
    assert all(request["model"] == "test-deployment" for request in requests)
    assert "explicit_citations: PASS" in result.stdout
    assert f"concise: {'FAIL' if fail_concise else 'PASS'}" in result.stdout
    assert URL in result.stdout
