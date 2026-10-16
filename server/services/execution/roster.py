"""Agent registry: one record per execution agent, persisted as JSON.

Replaces the old roster (a bare list of names). A record carries what is needed to find, reuse and retire
an agent: a stable id, the canonical name, aliases, a short description, the people and topics it handles,
a lifecycle status and activity counters.

Properties that matter:
  * Names are canonicalised, so "Email to Alice", "email to alice" and "Email to Alice!" are ONE agent
    (they used to be three roster entries sharing one log file).
  * Writes are atomic (temp file + os.replace), so a failed save can never leave an empty or partial file,
    and the module needs no Unix-only locking (it also imports on Windows).
  * A roster.json in the old format (a JSON list of names) is migrated on first load.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ...logging_config import logger
from .log_store import _slugify

ACTIVE = "active"
DORMANT = "dormant"
ARCHIVED = "archived"
MERGED = "merged"  # folded into another agent (see merged_into); its name is now an alias of that agent
REGISTRY_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_name(name: str) -> str:
    """Key under which an agent name is looked up: case, spacing and punctuation do not matter."""
    return _slugify(name or "")


@dataclass
class AgentRecord:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    people: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    status: str = ACTIVE
    created_at: str = field(default_factory=_now)
    last_active_at: str = field(default_factory=_now)
    run_count: int = 0
    merged_into: Optional[str] = None  # id of the agent this one was merged into

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        known = {f for f in cls.__dataclass_fields__}  # tolerate fields added by later versions
        return cls(**{k: v for k, v in data.items() if k in known})


class AgentRoster:
    """Registry of execution agents (kept under its historical name so callers do not change)."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._records: Dict[str, AgentRecord] = {}  # canonical key -> record
        self._lock = threading.RLock()
        self._defer_depth = 0
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        with self._lock:
            if not self._roster_path.exists():
                self._records = {}
                self.save()
                return
            try:
                raw = json.loads(self._roster_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._quarantine_corrupt_file(exc)
                self._records = {}
                return
            self._records = self._parse(raw)

    def _parse(self, raw) -> Dict[str, AgentRecord]:
        records: Dict[str, AgentRecord] = {}
        if isinstance(raw, list):  # old format: ["Email to Alice", ...]
            for name in raw:
                self._insert(records, AgentRecord(id=self._new_id(), name=str(name)))
            logger.info("Migrated roster.json from the old name-list format (%d agents)", len(records))
            return records
        if isinstance(raw, dict):
            for item in raw.get("agents", []):
                try:
                    self._insert(records, AgentRecord.from_dict(item))
                except Exception as exc:  # one bad record must not drop the others
                    logger.warning("Skipping unreadable agent record: %s", exc)
        return records

    def _quarantine_corrupt_file(self, exc: Exception) -> None:
        """Keep an unreadable file for recovery instead of letting the next save silently overwrite it."""
        backup = self._roster_path.with_name(f"{self._roster_path.name}.corrupt-{int(datetime.now().timestamp())}")
        try:
            os.replace(self._roster_path, backup)
            logger.error("roster.json unreadable (%s); kept a copy at %s", exc, backup.name)
        except Exception as move_exc:  # pragma: no cover - defensive
            logger.error("roster.json unreadable (%s) and could not be moved: %s", exc, move_exc)

    def save(self) -> None:
        """Write atomically: a crash or error leaves either the old file or the new one, never a partial one."""
        with self._lock:
            if self._defer_depth:
                return
            payload = {"version": REGISTRY_VERSION, "agents": [asdict(r) for r in self._records.values()]}
            try:
                self._roster_path.parent.mkdir(parents=True, exist_ok=True)
                handle, tmp_name = tempfile.mkstemp(dir=self._roster_path.parent, prefix=".roster-", suffix=".tmp")
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as tmp:
                        json.dump(payload, tmp, indent=2)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                    os.replace(tmp_name, self._roster_path)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_name)
                    raise
            except Exception as exc:
                logger.warning("Failed to save roster.json: %s", exc)

    @contextlib.contextmanager
    def batch(self) -> Iterator["AgentRoster"]:
        """Defer saving until the block ends (bulk seeding writes the file once, not once per agent)."""
        with self._lock:
            self._defer_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._defer_depth -= 1
            self.save()

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def _new_id() -> str:
        return "a_" + uuid.uuid4().hex[:8]

    @staticmethod
    def _insert(records: Dict[str, AgentRecord], record: AgentRecord) -> None:
        key = normalize_name(record.name)
        if key in records:  # two old entries that normalise alike are the same agent
            existing = records[key]
            if record.name != existing.name and record.name not in existing.aliases:
                existing.aliases.append(record.name)
            return
        records[key] = record

    def resolve(self, name: str) -> Optional[AgentRecord]:
        """Find an agent by canonical name or alias, ignoring case and punctuation. A merged agent resolves to
        the agent it was merged into."""
        key = normalize_name(name)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = next((r for r in self._records.values()
                               if r.status != MERGED and any(normalize_name(a) == key for a in r.aliases)), None)
            return self._follow_merges(record)

    def _follow_merges(self, record: Optional[AgentRecord]) -> Optional[AgentRecord]:
        seen = set()
        while record is not None and record.status == MERGED and record.merged_into and record.id not in seen:
            seen.add(record.id)
            record = self.get_by_id(record.merged_into)
        return record

    def get_by_id(self, agent_id: str) -> Optional[AgentRecord]:
        with self._lock:
            return next((r for r in self._records.values() if r.id == agent_id), None)

    def get_record(self, name: str) -> Optional[AgentRecord]:
        return self.resolve(name)

    def list_records(self, *, include_archived: bool = False) -> List[AgentRecord]:
        """Agents that can be used. Merged agents are never listed (their names resolve to the survivor)."""
        with self._lock:
            return [r for r in self._records.values()
                    if r.status != MERGED and (include_archived or r.status != ARCHIVED)]

    def get_agents(self) -> List[str]:
        """Canonical names of every agent that is not archived."""
        return [r.name for r in self.list_records()]

    # ------------------------------------------------------------------ changes
    def add_agent(
        self,
        agent_name: str,
        *,
        description: str = "",
        people: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
    ) -> AgentRecord:
        """Return the agent for this name, creating it only if no agent (or alias) matches."""
        with self._lock:
            record = self.resolve(agent_name)
            changed = False
            if record is None:
                record = AgentRecord(id=self._new_id(), name=agent_name.strip() or "agent")
                self._records[normalize_name(record.name)] = record
                changed = True
            elif record.status == ARCHIVED:
                record.status = ACTIVE  # asking for an archived agent brings it back
                changed = True
            if description and not record.description:
                record.description, changed = description, True
            for target, new_values in ((record.people, people), (record.topics, topics)):
                for value in new_values or []:
                    if value not in target:
                        target.append(value)
                        changed = True
            if changed:
                self.save()
            return record

    def touch(self, agent_name: str) -> None:
        """Record that the agent was just used."""
        with self._lock:
            record = self.resolve(agent_name)
            if record is None:
                return
            record.last_active_at = _now()
            record.run_count += 1
            if record.status == DORMANT:
                record.status = ACTIVE
            self.save()

    def archive_agent(self, agent_name: str) -> bool:
        """Retire one agent without touching the others. Its log stays on disk."""
        with self._lock:
            record = self.resolve(agent_name)
            if record is None:
                return False
            record.status = ARCHIVED
            self.save()
            return True

    def add_alias(self, agent_name: str, alias: str) -> Optional[AgentRecord]:
        """Let another name reach this agent."""
        with self._lock:
            record = self.resolve(agent_name)
            if record is None:
                return None
            if normalize_name(alias) != normalize_name(record.name) and alias not in record.aliases:
                record.aliases.append(alias)
                self.save()
            return record

    def mark_merged(self, record: AgentRecord, into: AgentRecord) -> None:
        """Fold `record` into `into`: its names and contacts move over, and it disappears from the list."""
        with self._lock:
            for name in [record.name, *record.aliases]:
                if normalize_name(name) != normalize_name(into.name) and name not in into.aliases:
                    into.aliases.append(name)
            for target, values in ((into.people, record.people), (into.topics, record.topics)):
                target.extend(v for v in values if v not in target)
            into.run_count += record.run_count
            into.last_active_at = max(into.last_active_at, record.last_active_at)
            record.status, record.merged_into = MERGED, into.id
            self.save()

    def set_status(self, agent_name: str, status: str) -> None:
        with self._lock:
            record = self.resolve(agent_name)
            if record is not None and record.status != status:
                record.status = status
                self.save()

    def clear(self) -> None:
        """Remove every agent (used by the app's reset)."""
        with self._lock:
            self._records = {}
            try:
                self._roster_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to clear roster.json: %s", exc)
            logger.info("Cleared agent roster")


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
