"""Small helpers shared by the tests."""

from __future__ import annotations

import asyncio
import json

import pytest


def run(coro):
    """Run a coroutine to completion (the suite does not need pytest-asyncio)."""
    return asyncio.run(coro)


def tool_call(name: str, args, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
    }


def completion(content: str = "", tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0}}


def known_bug(issue: str, reason: str):
    """Mark a test that states the CORRECT behaviour, which the current code does not have yet.

    strict=True: once a fix makes it pass, pytest fails the run ("XPASS(strict)") until the marker
    is removed, so a fixed bug can never silently stay marked as broken.
    """
    return pytest.mark.xfail(strict=True, reason=f"{issue}: {reason}")
