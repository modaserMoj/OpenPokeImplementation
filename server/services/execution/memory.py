"""Bounded, lossless memory for execution agents (issue 3).

Before: every run pasted the agent's WHOLE log into its prompt (a worker grew from 3.5k to 23.8k tokens over
50 runs), while tool results were cut to 500 characters, often mid-JSON.

Now:
  * The raw log stays append-only on disk (the archive). Only the PROMPT VIEW is bounded.
  * Prompt view = a rolling summary of older runs + the identifiers they mentioned + the most recent runs
    verbatim, never more than HISTORY_BUDGET_CHARS (about 2,000 tokens) whatever happens.
  * When enough older runs pile up, one summarisation call folds them into the summary (in the background,
    after a run finishes). If it fails, nothing is lost and the view is still bounded; it simply retries later.
  * Identifiers (draft/thread/message ids, email addresses) are extracted by code, not by the summariser, so
    they cannot be dropped or invented.
  * Tool results: the full output is saved in a side file; the log keeps a short rule-based digest plus the
    result id, and the agent can fetch the full output again with the get_tool_result tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...logging_config import logger
from .log_store import ExecutionAgentLogStore, _slugify, get_execution_agent_logs

HISTORY_BUDGET_CHARS = 7_000   # about 2,000 tokens for summary + identifiers + recent runs
SUMMARY_MAX_CHARS = 3_500      # about 1,000 tokens
RECENT_RUNS = 3                # always kept word for word
COMPACT_AFTER_RUNS = 5         # fold once this many runs have piled up beyond the recent ones
MAX_IDENTIFIERS = 40
DIGEST_MAX_CHARS = 600
ARGUMENTS_MAX_CHARS = 2_000
KEEP_RESULTS = 50              # full tool results kept per agent (agents with an active reminder keep all)
SUMMARISER_INPUT_MAX_CHARS = 60_000
SUMMARISER_OUTPUT_BUDGETS = (2_000, 4_000)  # max_tokens per attempt (reasoning models spend part of it thinking)

AUTO_COMPACT = True            # tests switch this off and call compact() directly

Entry = Tuple[str, str, str]   # (tag, timestamp, payload)


# --------------------------------------------------------------------------- runs
def split_runs(entries: Iterable[Entry]) -> List[List[Entry]]:
    """Group log entries into runs. A run starts at a request, or at the first entry after a final answer
    (older logs and reminder runs did not always record their request)."""
    runs: List[List[Entry]] = []
    current: List[Entry] = []
    for entry in entries:
        tag = entry[0]
        starts_new = tag == "agent_request" or (current and current[-1][0] == "agent_response")
        if starts_new and current:
            runs.append(current)
            current = []
        current.append(entry)
    if current:
        runs.append(current)
    return runs


def render_entries(entries: Sequence[Entry]) -> str:
    parts = []
    for tag, timestamp, payload in entries:
        safe = escape(payload, quote=False)
        parts.append(f'<{tag} timestamp="{timestamp}">{safe}</{tag}>' if timestamp else f"<{tag}>{safe}</{tag}>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- identifiers
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Named ids are kept whatever their length (a reminder id is often "1"); a bare "id" needs 3+ characters,
# otherwise list indexes and counters would flood the identifiers.
_ID_FIELD = re.compile(
    r'["\']?(draft_id|draftId|thread_id|threadId|message_id|messageId|trigger_id|proposal_id)["\']?\s*[:=]\s*["\']?([\w-]+)'
    r'|["\']?\b(id)["\']?\s*[:=]\s*["\']?([\w-]{3,})'
)


def extract_identifiers(text: str) -> List[str]:
    """Ids and email addresses mentioned in text, in order of appearance, without duplicates."""
    found: List[str] = []
    for match in _ID_FIELD.finditer(text or ""):
        key, value = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        item = f"{key}={value}"
        if item not in found:
            found.append(item)
    for address in _EMAIL.findall(text or ""):
        if address.lower() not in found:
            found.append(address.lower())
    return found


def _merge_identifiers(old: Sequence[str], new: Sequence[str]) -> List[str]:
    merged = [i for i in old if i not in new] + list(new)  # most recent last
    return merged[-MAX_IDENTIFIERS:]


# --------------------------------------------------------------------------- tool results
_ITEM_FIELDS = ("id", "messageId", "threadId", "from", "sender", "to", "subject", "date", "messageTimestamp", "name")


def _first_item_list(data: Any, depth: int = 0) -> Optional[List[dict]]:
    if depth > 3:
        return None
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data[:3]):
        return data
    if isinstance(data, dict):
        for value in data.values():
            found = _first_item_list(value, depth + 1)
            if found:
                return found
    return None


def digest_result(result_text: str) -> str:
    """A short, rule-based description of a tool result (no model call, so nothing is invented)."""
    if len(result_text) <= DIGEST_MAX_CHARS:
        return result_text
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return result_text[: DIGEST_MAX_CHARS - 1] + "…"
    parts: List[str] = []
    items = _first_item_list(data)
    if items:
        parts.append(f"{len(items)} items")
        for item in items[:10]:
            fields = [f"{k}={str(item[k])[:80]}" for k in _ITEM_FIELDS if k in item and item[k] not in (None, "")]
            if fields:
                parts.append("(" + ", ".join(fields) + ")")
    elif isinstance(data, dict):
        parts.append("keys: " + ", ".join(list(data)[:15]))
    ids = extract_identifiers(result_text)[:15]
    text = "; ".join(parts) + (" | ids: " + ", ".join(ids) if ids else "")
    return text[:DIGEST_MAX_CHARS]


class ToolResultStore:
    """Full tool outputs, one JSON file each, under <agent log dir>/results/<agent>/."""

    def __init__(self, logs: Optional[ExecutionAgentLogStore] = None):
        self._logs = logs

    def _dir(self, agent_name: str) -> Path:
        logs = self._logs or get_execution_agent_logs()
        return logs._base_dir / "results" / _slugify(agent_name)

    def save(self, agent_name: str, tool_name: str, arguments: str, result_text: str) -> str:
        result_id = "r_" + uuid.uuid4().hex[:8]
        folder = self._dir(agent_name)
        folder.mkdir(parents=True, exist_ok=True)
        record = {"id": result_id, "tool": tool_name, "arguments": arguments, "result": result_text,
                  "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (folder / f"{result_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return result_id

    def load(self, agent_name: str, result_id: str) -> Optional[dict]:
        if not re.fullmatch(r"r_[0-9a-f]{8}", result_id or ""):
            return None  # never build a path from an arbitrary string
        path = self._dir(agent_name) / f"{result_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def prune(self, agent_name: str, keep: int = KEEP_RESULTS) -> int:
        """Delete all but the newest `keep` results. Agents with an active reminder keep everything."""
        if agent_has_active_reminder(agent_name):
            return 0
        files = sorted(self._dir(agent_name).glob("r_*.json"), key=lambda p: p.stat().st_mtime)
        removed = 0
        for path in files[: max(0, len(files) - keep)]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


def agent_has_active_reminder(agent_name: str) -> bool:
    try:
        from ..triggers import get_trigger_service

        return any(t.status == "active" for t in get_trigger_service().list_triggers(agent_name=agent_name))
    except Exception:  # if in doubt, keep the data
        return True


def record_tool_call(agent_name: str, tool_name: str, arguments: str, result_text: str) -> str:
    """Log a tool call: full arguments (bounded), a digest of the result, and the full result in a side file."""
    logs = get_execution_agent_logs()
    args = arguments if len(arguments) <= ARGUMENTS_MAX_CHARS else arguments[:ARGUMENTS_MAX_CHARS] + "… [arguments shortened]"
    logs.record_action(agent_name, f"Calling {tool_name} with: {args}")
    if len(result_text) <= DIGEST_MAX_CHARS:
        logs.record_tool_response(agent_name, tool_name, result_text)
        return ""
    store = ToolResultStore()
    result_id = store.save(agent_name, tool_name, arguments, result_text)
    store.prune(agent_name)
    logs.record_tool_response(
        agent_name, tool_name,
        f"{digest_result(result_text)} [summary only; full result: {result_id}, {len(result_text):,} characters, "
        f"fetch with get_tool_result]",
    )
    return result_id


# --------------------------------------------------------------------------- summary state
@dataclass
class MemoryState:
    summary: str = ""
    identifiers: List[str] = field(default_factory=list)
    covered_runs: int = 0
    updated_at: str = ""


def _state_path(agent_name: str) -> Path:
    return get_execution_agent_logs()._base_dir / f"{_slugify(agent_name)}.memory.json"


def load_state(agent_name: str) -> MemoryState:
    try:
        data = json.loads(_state_path(agent_name).read_text(encoding="utf-8"))
        return MemoryState(**{k: v for k, v in data.items() if k in MemoryState.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return MemoryState()


def save_state(agent_name: str, state: MemoryState) -> None:
    path = _state_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".memory-", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(asdict(state), out)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- prompt view
def render_history(agent_name: str, *, budget: int = HISTORY_BUDGET_CHARS) -> str:
    """What the agent sees of its past: summary + identifiers + as many recent runs as fit. Never over budget."""
    runs = split_runs(get_execution_agent_logs().iter_entries(agent_name))
    if not runs:
        return ""
    state = load_state(agent_name)
    covered = min(state.covered_runs, len(runs))
    uncovered = runs[covered:]

    head: List[str] = []
    if state.summary:
        head.append(f"## Summary of earlier work ({covered} earlier runs)\n{state.summary[:SUMMARY_MAX_CHARS]}")
    if state.identifiers:
        head.append("Identifiers from earlier runs: " + ", ".join(state.identifiers))
    head_text = "\n\n".join(head)

    room = budget - len(head_text) - 40
    shown: List[str] = []
    for run in reversed(uncovered):
        text = render_entries(run)
        if len(text) > room:
            if not shown:  # the latest run alone is too big: keep its end
                shown.insert(0, "… (start of this run omitted)\n" + text[-max(room - 40, 0):])
            break
        shown.insert(0, text)
        room -= len(text) + 1
    omitted = len(uncovered) - len(shown)

    sections = [head_text] if head_text else []
    if omitted:
        sections.append(f"({omitted} older runs not shown here; they will be added to the summary)")
    if shown:
        sections.append("## Recent runs (oldest first)\n" + "\n".join(shown))
    return "\n\n".join(sections)


# --------------------------------------------------------------------------- compaction
Summariser = Callable[[str, str, str], Awaitable[str]]


async def summarise_runs(agent_name: str, previous_summary: str, runs_text: str) -> str:
    """Fold new activity into the agent's summary with one model call (the configured summariser model)."""
    from ...config import get_settings
    from ...openrouter_client import request_chat_completion
    from ...openrouter_client.usage import set_tag

    settings = get_settings()
    system = (
        f"You keep the working memory of an assistant's background worker called '{agent_name}'. "
        "Merge the previous summary with the new activity into ONE updated summary. Keep: what the worker is "
        "for, who it deals with, what was done and decided, what the user confirmed or rejected, and anything "
        "still open. Copy ids, email addresses, dates and amounts exactly. Do not invent anything. Drop chatter "
        f"and repetition. Plain text, at most about 350 words, sections: Purpose / Done / Open / Key facts."
    )
    user = f"PREVIOUS SUMMARY:\n{previous_summary or '(none)'}\n\nNEW ACTIVITY (oldest first):\n{runs_text}"
    set_tag(f"memory:{agent_name}")
    # Some models (Haiku 5.5) reason before answering and those tokens count against max_tokens: a tight cap
    # silently cut the summary after its first section. So: a generous budget, and a cut-off answer is retried
    # once with more room - and never saved.
    for budget in SUMMARISER_OUTPUT_BUDGETS:
        response = await request_chat_completion(
            model=settings.summarizer_model, messages=[{"role": "user", "content": user}], system=system,
            api_key=settings.openrouter_api_key, max_tokens=budget,
        )
        choice = (response.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if choice.get("finish_reason") != "length" and text:
            return text
    raise RuntimeError("summariser output was cut off or empty")


_locks: Dict[Tuple[int, str], asyncio.Lock] = {}


def _lock_for(agent_name: str) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), _slugify(agent_name))  # a lock belongs to one event loop
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


def needs_compaction(agent_name: str) -> bool:
    runs = split_runs(get_execution_agent_logs().iter_entries(agent_name))
    state = load_state(agent_name)
    uncovered = runs[min(state.covered_runs, len(runs)):]
    if len(uncovered) <= RECENT_RUNS:
        return False
    too_many = len(uncovered) >= RECENT_RUNS + COMPACT_AFTER_RUNS
    too_big = len(state.summary) + sum(len(render_entries(r)) for r in uncovered) > HISTORY_BUDGET_CHARS
    return too_many or too_big


async def compact(agent_name: str, summariser: Optional[Summariser] = None) -> bool:
    """Fold every run except the most recent ones into the summary. Returns True if it did."""
    async with _lock_for(agent_name):
        if not needs_compaction(agent_name):
            return False
        runs = split_runs(get_execution_agent_logs().iter_entries(agent_name))
        state = load_state(agent_name)
        start = min(state.covered_runs, len(runs))
        to_fold = runs[start:-RECENT_RUNS]
        text = "\n".join(render_entries(r) for r in to_fold)
        if len(text) > SUMMARISER_INPUT_MAX_CHARS:
            text = "… (oldest part omitted)\n" + text[-SUMMARISER_INPUT_MAX_CHARS:]
        try:
            summary = await (summariser or summarise_runs)(agent_name, state.summary, text)
        except Exception as exc:  # nothing is lost: the log is intact and the view stays bounded
            logger.warning("Memory compaction for %s failed, will retry later: %s", agent_name, exc)
            return False
        state.summary = summary.strip()[:SUMMARY_MAX_CHARS]
        # Identifiers come from what the agent asked, did and reported - not from raw tool output (a single email
        # search lists dozens of message ids, which would push out the draft/thread/reminder ids that matter;
        # tool output stays retrievable in full anyway).
        acted_on = "\n".join(payload for run in to_fold for tag, _, payload in run if tag != "tool_response")
        state.identifiers = _merge_identifiers(state.identifiers, extract_identifiers(acted_on))
        state.covered_runs = start + len(to_fold)
        state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_state(agent_name, state)
        logger.info("Compacted memory of %s: %d runs folded into the summary", agent_name, len(to_fold))
        return True


_pending: Set[asyncio.Task] = set()


def schedule_compaction(agent_name: str) -> None:
    """Compact in the background after a run, so the user never waits for it."""
    if not AUTO_COMPACT:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(compact(agent_name), name=f"compact-{agent_name}")
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def wait_for_compactions() -> None:
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
