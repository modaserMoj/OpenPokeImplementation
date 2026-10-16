"""The agent registry (replaces the plain list of names): identity, migration, lifecycle, safe writes."""

import asyncio
import json

from helpers import run

from server.agents.interaction_agent import tools as interaction_tools
from server.services.execution import AgentRoster, get_agent_roster, get_execution_agent_logs
from server.services.execution import roster as roster_module


def test_names_that_differ_only_by_case_or_punctuation_are_one_agent():
    roster = get_agent_roster()
    first = roster.add_agent("Email to Alice")
    second = roster.add_agent("email to alice")
    third = roster.add_agent("Email to Alice!")
    assert first.id == second.id == third.id
    assert roster.get_agents() == ["Email to Alice"]  # the first spelling stays the canonical name


def test_an_alias_resolves_to_its_agent():
    roster = get_agent_roster()
    record = roster.add_agent("Email to Alice")
    record.aliases.append("Alice reply")
    assert roster.resolve("alice REPLY").id == record.id


def test_metadata_is_stored_and_survives_a_reload():
    roster = get_agent_roster()
    roster.add_agent("Lease thread", description="Renewal of the Elm St lease", people=["Dana Cruz"], topics=["lease"])
    reloaded = AgentRoster(roster._roster_path).get_record("Lease thread")
    assert (reloaded.description, reloaded.people, reloaded.topics) == (
        "Renewal of the Elm St lease", ["Dana Cruz"], ["lease"])
    assert reloaded.id.startswith("a_") and reloaded.status == "active"


def test_an_old_name_list_roster_is_migrated(tmp_path):
    path = tmp_path / "old_roster.json"
    path.write_text(json.dumps(["Email to Alice", "Q3 Budget", "email to alice"]))
    roster = AgentRoster(path)
    assert roster.get_agents() == ["Email to Alice", "Q3 Budget"]
    assert [r.id.startswith("a_") for r in roster.list_records()] == [True, True]


def test_an_unreadable_file_is_kept_for_recovery_not_overwritten(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text("{ this is not json")
    roster = AgentRoster(path)
    assert roster.get_agents() == []
    kept = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert len(kept) == 1 and kept[0].read_text() == "{ this is not json"


def test_a_save_leaves_no_temp_files_behind(tmp_path):
    folder = tmp_path / "registry"
    roster = AgentRoster(folder / "roster.json")
    roster.add_agent("Email to Alice")
    assert sorted(p.name for p in folder.iterdir()) == ["roster.json"]


def test_batch_writes_the_file_once(tmp_path, monkeypatch):
    roster = AgentRoster(tmp_path / "roster.json")
    writes = []
    real_replace = roster_module.os.replace
    monkeypatch.setattr(roster_module.os, "replace", lambda a, b: (writes.append(b), real_replace(a, b))[1])
    with roster.batch():
        for i in range(20):
            roster.add_agent(f"Agent {i}")
    assert len(writes) == 1 and len(roster.get_agents()) == 20


def test_archiving_hides_an_agent_but_keeps_its_log_and_asking_for_it_brings_it_back():
    roster = get_agent_roster()
    roster.add_agent("Old thread")
    get_execution_agent_logs().record_request("Old thread", "hello")
    assert roster.archive_agent("Old thread") is True and roster.get_agents() == []
    assert list(get_execution_agent_logs().iter_entries("Old thread"))  # history is kept
    assert roster.add_agent("old thread").status == "active"
    assert roster.get_agents() == ["Old thread"]


def test_touch_counts_use_and_wakes_a_dormant_agent():
    roster = get_agent_roster()
    record = roster.add_agent("Daily")
    record.status = "dormant"
    roster.touch("Daily")
    assert record.run_count == 1 and record.status == "active"


def test_send_message_to_agent_reuses_the_agent_whatever_the_spelling(monkeypatch):
    ran = []

    async def fake_execute(name, instructions):
        ran.append(name)
        return type("Result", (), {"success": True})()

    monkeypatch.setattr(interaction_tools._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute)

    async def scenario():
        first = interaction_tools.send_message_to_agent("Email to Alice", "draft hello")
        second = interaction_tools.send_message_to_agent("email to alice", "also say thanks")
        await asyncio.sleep(0.05)
        return first, second

    first, second = run(scenario())
    assert first.payload["new_agent_created"] is True
    # The second call came before the first had created the agent; it is resolved before running, either way.
    assert ran == ["Email to Alice", "Email to Alice"]
    assert get_agent_roster().get_agents() == ["Email to Alice"]
    assert get_agent_roster().get_record("Email to Alice").run_count == 2  # one agent, used twice
