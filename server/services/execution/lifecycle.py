"""Agent lifecycle (issue 2): reuse instead of duplicating, merge duplicates that exist, and retire idle agents.

Three parts, one rule: A WRONG MERGE IS WORSE THAN A DUPLICATE (it would hand one person's context to another),
so every decision needs positive evidence and falls back to "keep separate".

  1. find_duplicate()  - when the main agent names a NEW agent, look for an existing agent that already covers it
                         (runs inline, before the work starts).
  2. consolidate()     - group near-duplicates that already exist; the most used one survives, the others become
                         aliases of it, their reminders move to it, their memory is noted in it. Nothing is deleted.
  3. apply_aging()     - idle 14 days -> dormant (left out of "recent", still searchable); idle 60 days -> archived
                         (hidden; asking for it by name brings it back). Never while the agent has an active reminder.

2 and 3 run as a background job (MaintenanceJob): shortly after startup, then every 24 hours.

When are two agents the same thread?
  * About a person: the SAME person (same email, same full name, or a first name only one contact has), and the
    same topic (embedding similarity, or shared topic words) - unless this is the only agent for that person.
    Two agents about different people are never merged, whatever the topic.
  * About a topic only: very high similarity (or, without embeddings, most of their topic words in common).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...logging_config import logger
from .retrieval import BM25Ranker, default_rankers, describe_request, extract_people, reciprocal_rank_fusion, tokenize
from .roster import ACTIVE, ARCHIVED, DORMANT, AgentRecord, AgentRoster, get_agent_roster

DORMANT_AFTER_DAYS = 14
ARCHIVE_AFTER_DAYS = 60
CANDIDATES = 8

# Similarity thresholds (cosine of unit vectors). Calibrated on the eval dataset with the configured embedding
# model (evals/eval/lifecycle_eval.py); chosen for zero false merges first, then as many true merges as possible.
SAME_TOPIC_SIMILARITY = 0.55      # same person, several agents for them: also needs this (or shared topic words)
DIFFERENT_TOPIC_SIMILARITY = 0.25  # same person but clearly another subject: never merge
TOPIC_ONLY_SIMILARITY = 0.80      # agents about a topic and no person
TOPIC_ONLY_NO_SHARED_WORD_SIMILARITY = 0.90  # ... and when they have no topic word in common
TOPIC_ONLY_WORD_OVERLAP = 0.5     # without embeddings: share of topic words in common (Jaccard)


# --------------------------------------------------------------------------- people
def _people(record_or_text) -> List[str]:
    if isinstance(record_or_text, AgentRecord):
        found = list(record_or_text.people) + extract_people(record_or_text.name)
    else:
        found = extract_people(record_or_text)
    return list(dict.fromkeys(p.strip() for p in found if p.strip()))


def _first(name: str) -> str:
    return name.split()[0].lower() if "@" not in name else name.lower()


@dataclass
class _Directory:
    """Who is who across the registry: which first names belong to more than one person."""

    first_name_owners: Dict[str, Set[str]] = field(default_factory=dict)
    agents_per_person: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, records: Sequence[AgentRecord]) -> "_Directory":
        directory = cls()
        for record in records:
            for person in _people(record):
                if "@" in person:
                    continue
                directory.first_name_owners.setdefault(_first(person), set()).add(person.lower())
                directory.agents_per_person.setdefault(_first(person), set()).add(record.id)
        return directory

    def unique_first_name(self, first: str) -> bool:
        owners = self.first_name_owners.get(first, set())
        full_names = {o for o in owners if len(o.split()) > 1}
        return len(full_names) <= 1


def same_person(a: Sequence[str], b: Sequence[str], directory: _Directory) -> bool:
    """True only with positive evidence that both sides talk about the same person."""
    a_l, b_l = {x.lower() for x in a}, {x.lower() for x in b}
    if any("@" in x and x in b_l for x in a_l):
        return True
    full_a = {x for x in a_l if "@" not in x and len(x.split()) > 1}
    full_b = {x for x in b_l if "@" not in x and len(x.split()) > 1}
    if full_a & full_b:
        return True
    # A full name on each side with the same first name but a different last name: two different people.
    for x in full_a:
        for y in full_b:
            if _first(x) == _first(y) and x != y:
                return False
    firsts_a = {_first(x) for x in a_l if "@" not in x}
    firsts_b = {_first(x) for x in b_l if "@" not in x}
    # A single name on one side that is the LAST name of exactly one known person on the other ("any word from
    # Lewis?" and Zeke Lewis), and nobody uses it as a first name.
    known_full = {o for owners in directory.first_name_owners.values() for o in owners if len(o.split()) > 1}
    for singles, fulls in ((a_l, full_b), (b_l, full_a)):
        for single in (x for x in singles if "@" not in x and len(x.split()) == 1):
            matches = {f for f in known_full | fulls if f.split()[-1] == single}
            if len(matches) == 1 and next(iter(matches)) in fulls and single not in directory.first_name_owners:
                return True
    for f in firsts_a & firsts_b:
        # A full name on the new side against an agent that only knows the first name: could be anyone with
        # that first name, so it is not evidence (a wrong merge is worse than a duplicate).
        if any(_first(x) == f for x in full_a) and not any(_first(x) == f for x in full_b):
            continue
        # A first name alone is enough only if it belongs to one person - counting the full names on BOTH sides,
        # so "Tess Okafor" in a new request is not taken for the only Tess the registry knows (Tess Gibson).
        known = {o for o in directory.first_name_owners.get(f, set()) if len(o.split()) > 1}
        known |= {x for x in full_a | full_b if _first(x) == f}
        if len(known) <= 1:
            return True
    return False


# --------------------------------------------------------------------------- the decision
# Task words say what to DO, not what the thread is ABOUT ("research the passport renewal" and "research the
# insurance renewal" are different threads), so they never count as shared topic words.
_TASK_WORDS = frozenset(
    "take care research option status follow up check handle coordinate update next step steps plan look into "
    "sort out find deal get move forward renew renewal review reply answer".split()
)


def _topic_words(text: str, people: Sequence[str]) -> Set[str]:
    names = {t for p in people for t in tokenize(p)}
    return {t for t in tokenize(text) if t not in names and t not in _TASK_WORDS and not t.isdigit()}


def _record_text(record: AgentRecord) -> str:
    return " ".join([record.name, record.description, " ".join(record.topics)])


def is_same_thread(
    new_text: str,
    new_people: Sequence[str],
    candidate: AgentRecord,
    similarity: Optional[float],
    directory: _Directory,
) -> bool:
    candidate_people = _people(candidate)
    new_words = _topic_words(new_text, new_people)
    old_words = _topic_words(_record_text(candidate), candidate_people)
    shared = new_words & old_words

    if new_people or candidate_people:
        if not (new_people and candidate_people) or not same_person(new_people, candidate_people, directory):
            return False
        if similarity is not None and similarity < DIFFERENT_TOPIC_SIMILARITY:
            return False
        firsts = {_first(p) for p in candidate_people if "@" not in p}
        only_agent_for_person = all(len(directory.agents_per_person.get(f, set())) <= 1 for f in firsts) and firsts
        if only_agent_for_person:
            return True
        return bool(shared) or (similarity is not None and similarity >= SAME_TOPIC_SIMILARITY)

    if similarity is not None:
        # Short topic phrases can be close in meaning yet be different threads ("Garden irrigation" vs "Plant
        # watering"), so with no topic word in common the meaning must be near-identical.
        needed = TOPIC_ONLY_SIMILARITY if shared else TOPIC_ONLY_NO_SHARED_WORD_SIMILARITY
        return similarity >= needed
    union = new_words | old_words
    return bool(union) and len(shared) >= 2 and len(shared) / len(union) >= TOPIC_ONLY_WORD_OVERLAP


# --------------------------------------------------------------------------- candidates and similarity
def _semantic_ranker(rankers):
    return next((r for r in rankers if hasattr(r, "vectors")), None)


async def _candidates(query: str, records: Sequence[AgentRecord], rankers, exclude_id: Optional[str] = None):
    rankings = []
    for ranker in rankers:
        ranking = ranker.rank(query, records)
        rankings.append(await ranking if asyncio.iscoroutine(ranking) else ranking)
    fused = reciprocal_rank_fusion(rankings, weights=[getattr(r, "weight", 1.0) for r in rankers])
    return [r for r in fused if r.id != exclude_id][:CANDIDATES]


async def _similarities(text: str, candidates: Sequence[AgentRecord], rankers) -> List[Optional[float]]:
    semantic = _semantic_ranker(rankers)
    if semantic is None or not candidates:
        return [None] * len(candidates)
    try:
        from .embeddings import agent_text

        vectors = await semantic.vectors([text] + [agent_text(c) for c in candidates])
    except Exception as exc:  # no embeddings available: decide on people and words only
        logger.warning("Duplicate check without embeddings: %s", exc)
        return [None] * len(candidates)
    query = vectors[0]
    return [sum(a * b for a, b in zip(query, v)) for v in vectors[1:]]


# --------------------------------------------------------------------------- 1. on creation
async def find_duplicate(
    name: str,
    instructions: str,
    *,
    roster: Optional[AgentRoster] = None,
    rankers=None,
) -> Optional[AgentRecord]:
    """An existing agent that already covers this new agent name + first request, or None."""
    roster = roster or get_agent_roster()
    records = roster.list_records(include_archived=True)
    if not records:
        return None
    rankers = list(rankers if rankers is not None else default_rankers())
    text = f"{name}. {instructions}"
    new_people = extract_people(f"{name}. {instructions}")
    candidates = await _candidates(text, records, rankers)
    similarities = await _similarities(text, candidates, rankers)
    directory = _Directory.build(records)
    for candidate, similarity in zip(candidates, similarities):
        if is_same_thread(text, new_people, candidate, similarity, directory):
            return candidate
    return None


async def resolve_new_agent(name: str, instructions: str, *, roster: Optional[AgentRoster] = None, rankers=None) -> str:
    """The agent that should handle a request sent to a name nobody has used yet: an existing agent that already
    covers it (the new name becomes its alias), or a newly created one."""
    roster = roster or get_agent_roster()
    existing = roster.resolve(name)  # another request may have created it meanwhile
    if existing is not None:
        return existing.name
    duplicate = await find_duplicate(name, instructions, roster=roster, rankers=rankers)
    if duplicate is not None:
        roster.add_alias(duplicate.name, name)
        if duplicate.status != ACTIVE:
            roster.set_status(duplicate.name, ACTIVE)
        logger.info("Reusing agent %r for the new name %r (same thread)", duplicate.name, name)
        return duplicate.name
    record = roster.add_agent(name, description=describe_request(instructions), people=extract_people(instructions))
    return record.name


# --------------------------------------------------------------------------- 2. consolidation
@dataclass
class ConsolidationReport:
    groups: List[List[str]] = field(default_factory=list)  # [survivor, merged, merged, ...]
    reminders_moved: int = 0

    @property
    def merged(self) -> int:
        return sum(len(g) - 1 for g in self.groups)


def _survivor(group: Sequence[AgentRecord], order: Dict[str, int]) -> AgentRecord:
    """Most used first, then the oldest (so its history is the longest)."""
    return sorted(group, key=lambda r: (-r.run_count, r.created_at, order[r.id]))[0]


async def find_duplicate_groups(roster: AgentRoster, rankers=None) -> List[List[AgentRecord]]:
    records = roster.list_records(include_archived=True)
    rankers = list(rankers if rankers is not None else default_rankers())
    directory = _Directory.build(records)
    parent = {r.id: r.id for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    from .embeddings import agent_text

    for record in records:
        text = agent_text(record)  # the same text the semantic ranker embeds for this agent: no extra calls
        candidates = await _candidates(text, records, rankers, exclude_id=record.id)
        similarities = await _similarities(text, candidates, rankers)
        for candidate, similarity in zip(candidates, similarities):
            if is_same_thread(text, _people(record), candidate, similarity, directory):
                parent[find(record.id)] = find(candidate.id)

    groups: Dict[str, List[AgentRecord]] = {}
    for record in records:
        groups.setdefault(find(record.id), []).append(record)
    return [g for g in groups.values() if len(g) > 1]


def merge_group(roster: AgentRoster, group: Sequence[AgentRecord], order: Dict[str, int]) -> Tuple[List[str], int]:
    from ..triggers import get_trigger_service
    from . import memory

    survivor = _survivor(group, order)
    moved = 0
    notes: List[str] = []
    state = memory.load_state(survivor.name)
    for record in group:
        if record.id == survivor.id:
            continue
        moved += get_trigger_service().reassign_agent(record.name, survivor.name)
        other = memory.load_state(record.name)
        gist = other.summary or _last_answer(record.name)
        if gist:
            notes.append(f"Merged in from '{record.name}': {' '.join(gist.split())[:400]}")
        state.identifiers = memory._merge_identifiers(state.identifiers, other.identifiers)
        roster.mark_merged(record, survivor)
    if notes:
        state.summary = "\n".join([state.summary, *notes]).strip()[: memory.SUMMARY_MAX_CHARS]
        memory.save_state(survivor.name, state)
    if survivor.status != ACTIVE and any(r.status == ACTIVE for r in group):
        roster.set_status(survivor.name, ACTIVE)
    return [survivor.name] + [r.name for r in group if r.id != survivor.id], moved


def _last_answer(agent_name: str) -> str:
    from .log_store import get_execution_agent_logs

    answers = [p for tag, _, p in get_execution_agent_logs().iter_entries(agent_name) if tag == "agent_response"]
    return answers[-1] if answers else ""


async def consolidate(roster: Optional[AgentRoster] = None, rankers=None) -> ConsolidationReport:
    roster = roster or get_agent_roster()
    order = {r.id: i for i, r in enumerate(roster.list_records(include_archived=True))}
    report = ConsolidationReport()
    for group in await find_duplicate_groups(roster, rankers):
        names, moved = merge_group(roster, group, order)
        report.groups.append(names)
        report.reminders_moved += moved
        logger.info("Merged duplicate agents into %r: %s", names[0], names[1:])
    return report


# --------------------------------------------------------------------------- 3. aging
def _parse(timestamp: str) -> datetime:
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def apply_aging(roster: Optional[AgentRoster] = None, now: Optional[datetime] = None) -> Dict[str, int]:
    from .memory import agent_has_active_reminder

    roster = roster or get_agent_roster()
    now = now or datetime.now(timezone.utc)
    changes = {"dormant": 0, "archived": 0}
    with roster.batch():
        for record in roster.list_records():
            if agent_has_active_reminder(record.name):
                continue
            idle = now - _parse(record.last_active_at)
            if idle >= timedelta(days=ARCHIVE_AFTER_DAYS):
                record.status = ARCHIVED
                changes["archived"] += 1
            elif idle >= timedelta(days=DORMANT_AFTER_DAYS) and record.status == ACTIVE:
                record.status = DORMANT
                changes["dormant"] += 1
    return changes


# --------------------------------------------------------------------------- the background job
class MaintenanceJob:
    """Consolidation then aging: shortly after startup, then every `interval` (default 24 hours)."""

    def __init__(self, interval_seconds: Optional[float] = None, initial_delay_seconds: float = 60.0) -> None:
        hours = float(os.getenv("OPENPOKE_MAINTENANCE_HOURS", "24"))
        self.interval = interval_seconds if interval_seconds is not None else hours * 3600
        self.initial_delay = initial_delay_seconds
        self._task: Optional[asyncio.Task] = None
        self.runs = 0

    async def run_once(self) -> dict:
        report = await consolidate()
        aging = apply_aging()
        self.runs += 1
        summary = {"merged": report.merged, "groups": report.groups, "reminders_moved": report.reminders_moved, **aging}
        logger.info("Agent maintenance: %s", {k: v for k, v in summary.items() if k != "groups"})
        return summary

    async def _loop(self) -> None:
        await asyncio.sleep(self.initial_delay)
        while True:
            try:
                await self.run_once()
            except Exception as exc:  # a failed pass must never stop the next one
                logger.error("Agent maintenance failed: %s", exc)
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="agent-maintenance")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


_job: Optional[MaintenanceJob] = None


def get_maintenance_job() -> MaintenanceJob:
    global _job
    if _job is None:
        _job = MaintenanceJob()
    return _job
