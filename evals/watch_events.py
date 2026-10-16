"""Merge OpenPoke runtime activity into one append-only file: evals/runtime/events.log.

Sources (polled every 0.5s, new bytes only):
  evals/runtime/server.log        app/uvicorn log
  server/data/conversation/*.log     user <-> interaction agent turns
  server/data/execution_agents/*.log per-agent actions, tool results, responses
  server/data/**/*.json              roster / seen-store snapshots (on change)
  server/data/triggers.db            trigger rows (on change)
  server/data/llm_usage.jsonl        one line per LLM call: tokens, cost, latency

    python evals/watch_events.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "runtime" / "events.log"
DATA = ROOT / "server" / "data"
TEXT_SOURCES = [ROOT / "evals" / "runtime" / "server.log"]
TEXT_GLOBS = ["conversation/*.log", "execution_agents/*.log"]
JSON_GLOBS = ["execution_agents/*.json", "triggers*/*.json", "*.json", "**/*.json"]


def emit(out, source: str, text: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in text.splitlines():
        if line.strip():
            out.write(f"{ts} [{source}] {line}\n")
    out.flush()


def snapshot_triggers(db: Path) -> str:
    """Compact view of the triggers table (read-only) for change detection."""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
        rows = conn.execute(
            "SELECT id, agent_name, status, next_trigger, recurrence_rule, last_error, payload FROM triggers ORDER BY id"
        ).fetchall()
        conn.close()
    except Exception:
        return "[]"
    return json.dumps([
        {"id": r[0], "agent": r[1], "status": r[2], "next": r[3], "rrule": r[4], "error": r[5], "payload": (r[6] or "")[:80]}
        for r in rows
    ])


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[Path, int] = {}
    json_hash: dict[Path, str] = {}
    last_triggers = None
    usage_offset = 0
    usage_path = DATA / "llm_usage.jsonl"
    with OUT.open("a", encoding="utf-8") as out:
        emit(out, "watcher", "started")
        while True:
            paths = list(TEXT_SOURCES)
            if DATA.exists():
                for g in TEXT_GLOBS:
                    paths += sorted(DATA.glob(g))
            for p in paths:
                if not p.exists():
                    continue
                size = p.stat().st_size
                # first sight of a pre-existing data file: start from its current end
                start = offsets.setdefault(p, 0 if p in TEXT_SOURCES or p.stat().st_mtime > time.time() - 5 else size)
                if size < start:
                    start = 0
                if size > start:
                    with p.open("rb") as f:
                        f.seek(start)
                        chunk = f.read().decode("utf-8", errors="replace")
                    offsets[p] = size
                    label = p.parent.name + "/" + p.name if p.parent.name != "runtime" else p.name
                    emit(out, label, chunk)
            if DATA.exists():
                for p in sorted(set(DATA.glob("**/*.json"))):
                    h = hashlib.md5(p.read_bytes()).hexdigest()
                    if json_hash.get(p) not in (None, h):
                        emit(out, "state/" + str(p.relative_to(DATA)), "changed: " + p.read_text()[:600].replace("\n", " "))
                    json_hash[p] = h
            snap = snapshot_triggers(DATA / "triggers.db") if (DATA / "triggers.db").exists() else "[]"
            if snap != last_triggers:
                if last_triggers is not None:
                    emit(out, "triggers.db", snap)
                last_triggers = snap
            if usage_path.exists() and usage_path.stat().st_size > usage_offset:
                with usage_path.open("rb") as f:
                    f.seek(usage_offset)
                    chunk = f.read().decode("utf-8", errors="replace")
                usage_offset = usage_path.stat().st_size
                for line in chunk.splitlines():
                    try:
                        u = json.loads(line)
                        emit(out, "llm", f"{u['model']} in={u['prompt_tokens']} out={u['completion_tokens']} "
                                         f"${u['cost_usd']} {u['latency_s']}s tools={u['tool_calls']} err={u['error']}")
                    except Exception:
                        pass
            time.sleep(0.5)


if __name__ == "__main__":
    main()
