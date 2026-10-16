from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

from ..config import get_settings
from .usage import record_call

OpenRouterBaseURL = "https://openrouter.ai/api/v1"

# Without a cap the request asks for the model maximum (64k for Sonnet 4), and OpenRouter
# rejects it with 402 unless the balance can cover that worst case.
DEFAULT_MAX_TOKENS = int(os.getenv("OPENPOKE_MAX_TOKENS", "2048"))


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


# One connection pool per event loop, reused by every call: no new TCP + TLS handshake per model call.
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


def shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120))
        _clients[loop] = client
    return client


async def close_shared_clients() -> None:
    for client in list(_clients.values()):
        await client.aclose()


def supports_prompt_cache(model: str) -> bool:
    """Anthropic models (through OpenRouter) cache a prompt prefix marked with cache_control."""
    return (model or "").startswith("anthropic/")


def _text(text: str, cached: bool) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "text", "text": text}
    if cached:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _build_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, Sequence[str]]],
    *,
    cache: bool = False,
) -> List[Dict[str, Any]]:
    """Messages for the API. With `cache`, each system block and the opening user message end with a cache
    marker (at most 3 markers; Anthropic allows 4): the stable start of the prompt is then read from cache on the
    next call within about 5 minutes - faster, and about a tenth of the input price."""
    blocks = [system] if isinstance(system, str) else [b for b in (system or []) if b]
    out: List[Dict[str, Any]] = []
    if blocks:
        if cache:
            out.append({"role": "system", "content": [_text(b, True) for b in blocks[:2]]
                        + ([_text("\n\n".join(blocks[2:]), False)] if len(blocks) > 2 else [])})
        else:
            out.append({"role": "system", "content": "\n\n".join(blocks)})
    rest = list(messages)
    if cache and rest and rest[0].get("role") == "user" and isinstance(rest[0].get("content"), str):
        rest[0] = {**rest[0], "content": [_text(rest[0]["content"], True)]}
    return out + rest


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, Sequence[str]]] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
    max_tokens: Optional[int] = None,
    cache: bool = False,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload.

    `system` may be one string or several blocks (stable first). `cache=True` marks them for prompt caching when
    the model supports it."""

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system, cache=cache and supports_prompt_cache(model)),
        "stream": False,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url.rstrip('/')}/chat/completions"

    started = time.monotonic()
    client = shared_client()
    try:
        response = await client.post(
            url,
            headers=_headers(api_key=api_key),
            json=payload,
            timeout=60.0,  # Set reasonable timeout instead of None
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            record_call(model=model, latency_seconds=time.monotonic() - started, response=None,
                        tools_offered=len(tools or []), error=f"http {exc.response.status_code}")
            _handle_response_error(exc)
        data = response.json()
        record_call(model=model, latency_seconds=time.monotonic() - started, response=data,
                    tools_offered=len(tools or []))
        return data
    except httpx.HTTPStatusError as exc:  # pragma: no cover - handled above
        _handle_response_error(exc)
    except httpx.HTTPError as exc:
        record_call(model=model, latency_seconds=time.monotonic() - started, response=None,
                    tools_offered=len(tools or []), error=type(exc).__name__)
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    raise OpenRouterError("OpenRouter request failed: unknown error")


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL", "shared_client", "close_shared_clients"]
