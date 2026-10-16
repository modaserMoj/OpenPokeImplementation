"""Latency: slow Gmail calls no longer freeze the app, one connection is reused for model calls, and the stable
start of the Sonnet prompts is marked for prompt caching."""

import asyncio
import json
import time

import httpx
import pytest
from helpers import completion, run, tool_call

from server.openrouter_client import client as client_module

SLOW = 0.3


async def _ticks_while(coro, every=0.02):
    """Run `coro` and count how often another coroutine got to run meanwhile (0 = the event loop was blocked)."""
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(every)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await coro
    finally:
        done.set()
        await task
    return ticks


# ------------------------------------------------------------------ fix 1: no blocking on synchronous Gmail calls
def test_a_slow_gmail_tool_call_does_not_freeze_the_event_loop(llm, gmail, monkeypatch):
    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.agents.execution_agent.tools import gmail as gmail_tools

    def slow_execute(tool_name, user_id, arguments):
        time.sleep(SLOW)  # the Composio SDK is synchronous
        return {"successful": True, "data": {"id": "draft-x"}}

    monkeypatch.setattr(gmail_tools, "_execute", slow_execute)
    llm.script(completion("", [tool_call("gmail_create_draft", {"recipient_email": "a@b.test", "subject": "s", "body": "b"})]),
               completion("done"))
    ticks = run(_ticks_while(ExecutionAgentRuntime("Mailer").execute("draft it")))
    assert ticks >= 8, f"event loop was blocked ({ticks} ticks during a {SLOW}s Gmail call)"


def test_the_email_watcher_fetch_does_not_freeze_the_event_loop(monkeypatch):
    from server.services.gmail import importance_watcher

    def slow_fetch(*args, **kwargs):
        time.sleep(SLOW)
        return {"data": {"messages": []}}

    monkeypatch.setattr(importance_watcher, "execute_gmail_tool", slow_fetch)
    monkeypatch.setattr(importance_watcher, "get_active_gmail_user_id", lambda: "user-1")
    ticks = run(_ticks_while(importance_watcher.ImportantEmailWatcher()._poll_once()))
    assert ticks >= 8


def test_the_email_search_fetch_does_not_freeze_the_event_loop(monkeypatch):
    from server.agents.execution_agent.tasks.search_email import tool as search_tool

    def slow_fetch(*args, **kwargs):
        time.sleep(SLOW)
        return {"data": {"messages": []}}

    monkeypatch.setattr(search_tool, "execute_gmail_tool", slow_fetch)
    ticks = run(_ticks_while(search_tool._perform_search(arguments={"query": "from:alice"}, queries=[], emails={},
                                                         composio_user_id="user-1")))
    assert ticks >= 8


# ------------------------------------------------------------------ fix 2: one connection pool
_REAL_POST = httpx.AsyncClient.post


def test_model_calls_reuse_one_connection_pool(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPOKE_USAGE_LOG", str(tmp_path / "u.jsonl"))
    created = []
    real = httpx.AsyncClient

    class Counting(real):
        post = _REAL_POST

        def __init__(self, *args, **kwargs):
            created.append(1)
            kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {}}]}))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", Counting)

    async def three_calls():
        for _ in range(3):
            await client_module.request_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}], api_key="k")

    run(three_calls())
    assert len(created) == 1  # was one new client (and TLS handshake) per call


# ------------------------------------------------------------------ fix 4: prompt caching
def _payload(**kwargs):
    return client_module._build_messages(kwargs.pop("messages", [{"role": "user", "content": "turn"}]), **kwargs)


def test_cache_markers_on_the_system_prompt_and_opening_message():
    messages = _payload(system="SYSTEM PROMPT", cache=True)
    assert messages[0] == {"role": "system", "content": [
        {"type": "text", "text": "SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}]}
    assert messages[1]["content"] == [{"type": "text", "text": "turn", "cache_control": {"type": "ephemeral"}}]


def test_the_worker_prompt_is_cached_as_two_blocks():
    messages = _payload(system=["BASE", "MEMORY"], cache=True)
    assert [b["text"] for b in messages[0]["content"]] == ["BASE", "MEMORY"]
    assert all(b.get("cache_control") for b in messages[0]["content"])
    total = sum(1 for m in messages for b in (m["content"] if isinstance(m["content"], list) else []) if "cache_control" in b)
    assert total <= 4  # Anthropic allows at most 4 cache markers


def test_without_caching_the_request_is_exactly_as_before():
    assert _payload(system="S", cache=False) == [{"role": "system", "content": "S"}, {"role": "user", "content": "turn"}]
    assert _payload(system=["BASE", "MEMORY"], cache=False)[0] == {"role": "system", "content": "BASE\n\nMEMORY"}


def test_caching_is_only_requested_for_models_that_support_it():
    assert client_module.supports_prompt_cache("anthropic/claude-sonnet-5.5")
    assert not client_module.supports_prompt_cache("openai/gpt-5")


def _sent_payload(monkeypatch, tmp_path, model):
    monkeypatch.setenv("OPENPOKE_USAGE_LOG", str(tmp_path / "u.jsonl"))
    seen = {}
    real = httpx.AsyncClient

    class Capture(real):
        post = _REAL_POST

        def __init__(self, *args, **kwargs):
            def handler(request):
                seen.update(json.loads(request.content))
                return httpx.Response(200, json={"choices": [{"message": {}}], "usage": {
                    "prompt_tokens": 4000, "prompt_tokens_details": {"cached_tokens": 3500, "cache_write_tokens": 0}}})
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", Capture)
    run(client_module.request_chat_completion(model=model, messages=[{"role": "user", "content": "hi"}],
                                              system="SYS", api_key="k", cache=True))
    return seen


def test_an_anthropic_request_carries_the_markers_and_cached_tokens_are_logged(monkeypatch, tmp_path):
    seen = _sent_payload(monkeypatch, tmp_path, "anthropic/claude-sonnet-5.5")
    assert seen["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    entry = json.loads((tmp_path / "u.jsonl").read_text().splitlines()[0])
    assert entry["cached_tokens"] == 3500 and entry["cache_write_tokens"] == 0


def test_other_providers_get_plain_strings(monkeypatch, tmp_path):
    seen = _sent_payload(monkeypatch, tmp_path, "openai/gpt-5")
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}


def test_the_main_agent_and_the_worker_ask_for_caching(llm, gmail):
    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime

    llm.script(completion("ok"), completion(""))
    run(ExecutionAgentRuntime("Worker").execute("x"))
    run(InteractionAgentRuntime().execute("hi"))
    worker, main = llm.calls
    assert worker["cache"] is True and len(worker["system_blocks"]) >= 1
    assert main["cache"] is True
