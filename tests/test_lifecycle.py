"""Issue 2: reuse instead of duplicating, merge existing duplicates, retire idle agents - never a wrong merge."""

import asyncio
import re
from datetime import datetime, timedelta, timezone

import pytest
from helpers import run

from server.services.execution import AgentRoster, get_agent_roster, get_execution_agent_logs, lifecycle, memory
from server.services.execution.embeddings import EmbeddingRanker
from server.services.execution.retrieval import BM25Ranker, extract_people, select_agents
from server.services.execution.roster import ARCHIVED, DORMANT, MERGED

WORDS_ONLY = [BM25Ranker()]


class ConceptEmbedder:
    """Words that mean the same thing share a dimension, like an embedding model would place them."""
    CONCEPTS = [{"passport", "travel", "document"}, {"pet", "dog", "vaccination", "shots"},
                {"garden", "irrigation", "plant", "watering"}, {"lease", "rental", "apartment"}]

    async def embed(self, texts):
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            v = [float(len(words & c)) for c in self.CONCEPTS] + [0.05]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


def _semantic():
    return [BM25Ranker(weight=0.5), EmbeddingRanker(ConceptEmbedder())]


def _add(roster, name, request, **kw):
    return roster.add_agent(name, description=request, people=extract_people(request), **kw)


# ------------------------------------------------------------------ who is a person
@pytest.mark.parametrize("text, people", [
    ("follow up with Yuri Roberts on the reference letter", ["Yuri Roberts"]),
    ("reply to Alice's last message", ["Alice"]),
    ("email Tess Gibson and cc bob@example.test", ["bob@example.test", "Tess Gibson"]),
    ("Podcast Guest - Hannah", []),           # a Title Case topic is not a person
    ("Follow up on the Visa Documents", []),  # neither is "Follow"
])
def test_people_are_only_names_a_cue_marks_as_a_person(text, people):
    assert extract_people(text) == people


# ------------------------------------------------------------------ 1. the check on creation
@pytest.fixture
def registry(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    _add(roster, "Email to Alice", "email Alice Chen about the lease renewal")
    _add(roster, "Email to Bob", "email Bob Stone about the lease renewal")
    _add(roster, "Kitchen remodel", "take care of the kitchen remodel")
    _add(roster, "Passport renewal", "take care of the passport renewal")
    return roster


def _dup(registry, name, text, rankers=WORDS_ONLY):
    found = run(lifecycle.find_duplicate(name, text, roster=registry, rankers=rankers))
    return found.name if found else None


def test_a_new_name_for_the_same_person_and_thread_reuses_the_agent(registry):
    assert _dup(registry, "Alice reply", "reply to Alice about the lease") == "Email to Alice"


@pytest.mark.parametrize("name, text", [
    ("Email to Alice Wong", "email Alice Wong about the lease renewal"),  # same first name, another person
    ("Email to Carol", "email Carol Diaz about the lease renewal"),       # same topic, another person
    ("Violin lessons", "find violin lessons for the kids"),               # brand new
])
def test_anything_less_than_clear_evidence_creates_a_new_agent(registry, name, text):
    assert _dup(registry, name, text) is None


def test_a_topic_agent_is_reused_when_most_topic_words_match(registry):
    assert _dup(registry, "Kitchen remodel follow-up", "follow up on the kitchen remodel") == "Kitchen remodel"


def test_a_paraphrased_topic_needs_embeddings_and_a_shared_word_or_near_identical_meaning(registry):
    assert _dup(registry, "Travel document", "renew my travel document", WORDS_ONLY) is None
    assert _dup(registry, "Passport follow-up", "check on the passport", _semantic()) == "Passport renewal"


def test_resolving_a_new_name_reuses_and_remembers_it_as_an_alias(registry):
    name = run(lifecycle.resolve_new_agent("Alice reply", "reply to Alice about the lease", roster=registry, rankers=WORDS_ONLY))
    assert name == "Email to Alice" and registry.resolve("alice reply").name == "Email to Alice"
    created = run(lifecycle.resolve_new_agent("Violin lessons", "find violin lessons", roster=registry, rankers=WORDS_ONLY))
    assert created == "Violin lessons" and registry.get_record("Violin lessons") is not None


def test_send_message_to_agent_reuses_the_thread_for_a_new_name(monkeypatch):
    from server.agents.interaction_agent import tools as interaction_tools

    roster = get_agent_roster()
    _add(roster, "Email to Alice", "email Alice Chen about the lease renewal")
    ran = []

    async def fake_execute(name, instructions):
        ran.append(name)
        return type("R", (), {"success": True})()

    monkeypatch.setattr(interaction_tools._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute)

    async def scenario():
        result = interaction_tools.send_message_to_agent("Alice reply", "reply to Alice about the lease")
        await asyncio.sleep(0.05)
        return result

    result = run(scenario())
    assert result.payload["new_agent_created"] is True and "reused" in result.payload["note"]
    assert ran == ["Email to Alice"] and roster.get_agents() == ["Email to Alice"]


# ------------------------------------------------------------------ 2. consolidation
@pytest.fixture
def duplicates(isolated_triggers):
    roster = get_agent_roster()
    survivor = _add(roster, "Email to Alice", "email Alice Chen about the lease renewal")
    survivor.run_count = 5
    _add(roster, "Alice reply", "reply to Alice's message about the lease")
    _add(roster, "Follow up with Alice Chen", "follow up with Alice Chen on the lease renewal")
    _add(roster, "Email to Bob", "email Bob Stone about the lease renewal")  # same topic, other person: stays
    logs = get_execution_agent_logs()
    logs.record_request("Alice reply", "reply to Alice")
    logs.record_agent_response("Alice reply", "Replied to Alice, thread_id=t-77, she wants Tuesday.")
    start = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    isolated_triggers.create_trigger(agent_name="Alice reply", payload="chase Alice", start_time=start, timezone_name="UTC")
    return roster


def test_consolidation_merges_the_thread_and_keeps_other_people_apart(duplicates):
    report = run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    assert report.groups == [["Email to Alice", "Alice reply", "Follow up with Alice Chen"]]
    assert sorted(duplicates.get_agents()) == ["Email to Alice", "Email to Bob"]


def test_merged_names_still_reach_the_survivor(duplicates):
    run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    assert duplicates.resolve("Alice reply").name == "Email to Alice"
    assert duplicates.resolve("follow up with alice chen").name == "Email to Alice"
    assert duplicates.get_record("Email to Alice").run_count == 5
    assert all(r.status != MERGED for r in duplicates.list_records())


def test_reminders_move_to_the_survivor(duplicates, isolated_triggers):
    report = run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    assert report.reminders_moved == 1
    assert [t.payload for t in isolated_triggers.list_triggers(agent_name="Email to Alice")] == ["chase Alice"]
    assert isolated_triggers.list_triggers(agent_name="Alice reply") == []


def test_the_survivor_remembers_what_the_merged_agents_did_and_nothing_is_deleted(duplicates):
    run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    state = memory.load_state("Email to Alice")
    assert "Merged in from 'Alice reply'" in state.summary and "Tuesday" in state.summary
    assert list(get_execution_agent_logs().iter_entries("Alice reply"))  # its log is still on disk


def test_merged_agents_never_appear_in_the_list_the_main_agent_sees(duplicates):
    run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    shown = [r.name for r in select_agents(duplicates.list_records(), "lease")]
    assert "Alice reply" not in shown and "Email to Alice" in shown


def test_consolidation_twice_changes_nothing_the_second_time(duplicates):
    run(lifecycle.consolidate(duplicates, WORDS_ONLY))
    assert run(lifecycle.consolidate(duplicates, WORDS_ONLY)).merged == 0


# ------------------------------------------------------------------ 3. aging
def _idle(roster, name, days):
    roster.get_record(name).last_active_at = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


def test_idle_agents_become_dormant_then_archived(isolated_triggers):
    roster = get_agent_roster()
    for name, days in (("Fresh", 1), ("Quiet", 20), ("Ancient", 70)):
        roster.add_agent(name)
        _idle(roster, name, days)
    assert lifecycle.apply_aging(roster) == {"dormant": 1, "archived": 1}
    assert [roster.get_record(n).status for n in ("Fresh", "Quiet")] == ["active", DORMANT]
    assert roster.get_record("Ancient") is not None and roster.get_record("Ancient").status == ARCHIVED
    assert sorted(roster.get_agents()) == ["Fresh", "Quiet"]


def test_an_agent_with_an_active_reminder_never_ages(isolated_triggers):
    roster = get_agent_roster()
    roster.add_agent("Daily digest")
    _idle(roster, "Daily digest", 90)
    start = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    isolated_triggers.create_trigger(agent_name="Daily digest", payload="digest", start_time=start,
                                     recurrence_rule="FREQ=DAILY", timezone_name="UTC")
    lifecycle.apply_aging(roster)
    assert roster.get_record("Daily digest").status == "active"


def test_using_a_dormant_or_archived_agent_brings_it_back(isolated_triggers):
    roster = get_agent_roster()
    for name, days in (("Quiet", 20), ("Ancient", 70)):
        roster.add_agent(name)
        _idle(roster, name, days)
    lifecycle.apply_aging(roster)
    roster.touch("Quiet")
    roster.add_agent("ancient")
    assert roster.get_record("Quiet").status == "active" and roster.get_record("Ancient").status == "active"


# ------------------------------------------------------------------ the background job
def test_the_maintenance_job_runs_at_start_and_then_on_its_interval(monkeypatch):
    calls = []

    async def fake_consolidate(*a, **k):
        calls.append("consolidate")
        return lifecycle.ConsolidationReport()

    monkeypatch.setattr(lifecycle, "consolidate", fake_consolidate)
    monkeypatch.setattr(lifecycle, "apply_aging", lambda *a, **k: {"dormant": 0, "archived": 0})
    job = lifecycle.MaintenanceJob(interval_seconds=0.05, initial_delay_seconds=0)

    async def scenario():
        await job.start()
        await asyncio.sleep(0.18)
        await job.stop()

    run(scenario())
    assert job.runs >= 3 and calls.count("consolidate") == job.runs


def test_a_failed_pass_does_not_stop_the_job(monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(lifecycle, "consolidate", broken)
    job = lifecycle.MaintenanceJob(interval_seconds=0.03, initial_delay_seconds=0)
    attempts = []
    original = job.run_once

    async def counted():
        attempts.append(1)
        return await original()

    job.run_once = counted

    async def scenario():
        await job.start()
        await asyncio.sleep(0.12)
        await job.stop()

    run(scenario())
    assert len(attempts) >= 3


def test_a_surname_alone_matches_the_one_person_who_has_it(registry):
    _add(registry, "Message Zeke Lewis", "send Zeke Lewis a message about the dentist reschedule")
    assert extract_people("send Zeke Lewis a message") == ["Zeke Lewis"]
    assert _dup(registry, "Lewis dentist check", "check with Lewis about the dentist reschedule") == "Message Zeke Lewis"


def test_a_surname_shared_by_two_people_is_not_evidence(registry):
    _add(registry, "Message Zeke Lewis", "send Zeke Lewis a message about the dentist reschedule")
    _add(registry, "Message Ann Lewis", "send Ann Lewis a message about the dentist reschedule")
    assert _dup(registry, "Lewis dentist check", "check with Lewis about the dentist reschedule") is None


def test_the_list_line_shows_who_an_agent_deals_with_and_its_other_names():
    from server.agents.interaction_agent.agent import _format_agents

    roster = get_agent_roster()
    record = _add(roster, "Email to Zeke", "email Zeke about the dentist reschedule")
    roster.add_alias("Email to Zeke", "Message Zeke Lewis")
    record.people.append("Zeke Lewis")
    line = _format_agents([record], [record])
    assert 'people="Zeke, Zeke Lewis"' in line and 'also_called="Message Zeke Lewis"' in line



def test_an_address_at_the_end_of_a_sentence_has_no_trailing_period():
    """Found live: 'email jane.doe@example.com.' stored the address with the full stop, so it no longer
    matched the same address elsewhere."""
    assert extract_people("Draft an email to jane.doe@example.com.") == ["jane.doe@example.com"]
