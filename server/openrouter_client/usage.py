"""Append-only per-call LLM usage log (tokens, cost, latency) as JSON lines."""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging_config import logger

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_usage.jsonl"
_lock = threading.Lock()
_tag: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("llm_usage_tag", default=None)


def set_tag(tag: Optional[str]) -> None:
    """Label subsequent calls in this context (e.g. an eval request id)."""
    _tag.set(tag)


def usage_log_path() -> Path:
    return Path(os.getenv("OPENPOKE_USAGE_LOG") or _DEFAULT_PATH)


def record_call(
    *,
    model: str,
    latency_seconds: float,
    response: Optional[Dict[str, Any]],
    tools_offered: int,
    error: Optional[str] = None,
) -> None:
    """Write one usage line. Never raises: accounting must not break a request."""
    try:
        usage = (response or {}).get("usage") or {}
        message = (((response or {}).get("choices") or [{}])[0]).get("message") or {}
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": _tag.get(),
            "model": model,
            "latency_s": round(latency_seconds, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "cache_write_tokens": (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens"),
            "tools_offered": tools_offered,
            "tool_calls": [
                (tc.get("function") or {}).get("name") for tc in (message.get("tool_calls") or [])
            ],
            "error": error,
        }
        path = usage_log_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("usage logging failed: %s", exc)
