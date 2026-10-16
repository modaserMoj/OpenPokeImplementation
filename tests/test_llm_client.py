"""The OpenRouter client changes made during this work: output cap and per-call usage log."""

import json

import httpx
import pytest
from helpers import run

from server.openrouter_client import client as client_module
from server.openrouter_client import usage


_REAL_POST = httpx.AsyncClient.post  # captured before the autouse no-network guard replaces it


def _fake_transport(handler):
    real = httpx.AsyncClient

    class Patched(real):
        post = _REAL_POST  # goes through the in-memory MockTransport below, never the network

        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return Patched


@pytest.fixture
def usage_log(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENPOKE_USAGE_LOG", str(path))
    return path


def _call(monkeypatch, handler, **kwargs):
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _fake_transport(handler))
    return run(client_module.request_chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}], api_key="k", **kwargs))


def test_requests_carry_an_output_cap(monkeypatch, usage_log):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    _call(monkeypatch, handler)
    assert seen["max_tokens"] == client_module.DEFAULT_MAX_TOKENS <= 4096  # without it OpenRouter assumes 64k


def test_every_successful_call_is_logged_with_tokens_and_cost(monkeypatch, usage_log):
    body = {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "wait"}}]}}],
            "usage": {"prompt_tokens": 3300, "completion_tokens": 120, "cost": 0.0118}}
    _call(monkeypatch, lambda request: httpx.Response(200, json=body))
    entry = json.loads(usage_log.read_text().splitlines()[0])
    assert (entry["prompt_tokens"], entry["completion_tokens"], entry["cost_usd"]) == (3300, 120, 0.0118)
    assert entry["tool_calls"] == ["wait"] and entry["error"] is None and entry["latency_s"] >= 0


def test_failed_calls_are_logged_too(monkeypatch, usage_log):
    with pytest.raises(client_module.OpenRouterError):
        _call(monkeypatch, lambda request: httpx.Response(429, json={"error": "slow down"}))
    assert json.loads(usage_log.read_text().splitlines()[0])["error"] == "http 429"


def test_a_broken_usage_log_never_breaks_a_request(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPOKE_USAGE_LOG", str(tmp_path))  # a directory, so opening it for append fails
    result = _call(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    assert result["choices"][0]["message"]["content"] == "ok"


def test_tag_labels_the_next_calls(monkeypatch, usage_log):
    usage.set_tag("eval:exact-01")
    _call(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"message": {}}]}))
    usage.set_tag(None)
    assert json.loads(usage_log.read_text().splitlines()[0])["tag"] == "eval:exact-01"
