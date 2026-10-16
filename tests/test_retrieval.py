"""Agent retrieval: the main agent sees a short list of relevant agents, not every name (issue 1)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "eval"))
import dataset  # noqa: E402

from server.agents.interaction_agent.agent import _render_active_agents, prepare_message_with_history  # noqa: E402
from server.services.execution import AgentRoster, get_agent_roster  # noqa: E402
from server.services.execution.retrieval import (  # noqa: E402
    BM25Ranker, describe_request, extract_people, reciprocal_rank_fusion, select_agents, tokenize)

SHORTLIST = 10 + 3  # limit + recent


def _record(roster, name, description="", people=(), topics=()):
    return roster.add_agent(name, description=description, people=list(people), topics=list(topics))


def _dataset_roster(tmp_path):
    roster = AgentRoster(tmp_path / "dataset_roster.json")
    truth = dataset.ground_truth()
    with roster.batch():
        for a in truth["agents"]:
            roster.add_agent(a["name"], description=a["description"], people=a["people"], topics=a["topics"])
    return roster, truth


def test_tokenizer_drops_filler_and_light_stems():
    assert tokenize("Please email the Invoices for Tess, 2041!") == ["invoice", "tes", "2041"] or \
        tokenize("Please email the Invoices for Tess, 2041!") == ["invoice", "tess", "2041"]
    assert tokenize("") == []


def test_a_rare_name_beats_generic_words(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    target = _record(roster, "Email to Tess", "Emails with Tess Gibson about school pickup", ["Tess Gibson"])
    _record(roster, "Email to Bob", "Emails with Bob about school fundraiser", ["Bob Lee"])
    _record(roster, "Dentist", "Reschedule the dentist")
    ranked = BM25Ranker().rank("any word from Gibson on the school pickup?", roster.list_records())
    assert ranked[0].id == target.id


def test_agents_with_no_matching_word_are_left_out(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    _record(roster, "Tokyo trip", "Plan the Tokyo itinerary")
    assert BM25Ranker().rank("renew my passport", roster.list_records()) == []


def test_fusion_prefers_agents_both_rankers_like(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    a, b, c = (_record(roster, n) for n in ("A", "B", "C"))
    fused = reciprocal_rank_fusion([[a, b, c], [c, b, a]])
    assert [r.id for r in reciprocal_rank_fusion([[a, b, c], [b, c, a]])][0] == b.id  # high in both lists wins
    assert reciprocal_rank_fusion([[a, b], [b, a]], weights=[1.0, 0.1])[0].id == a.id  # weights shift the balance


def test_a_small_roster_is_shown_in_full(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    for i in range(SHORTLIST):
        _record(roster, f"Agent {i}", f"does thing {i}")
    assert len(select_agents(roster.list_records(), "anything")) == SHORTLIST


def test_a_large_roster_is_cut_to_a_short_list_that_keeps_recent_agents(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    with roster.batch():
        for i in range(300):
            roster.add_agent(f"Agent {i}", description=f"topic{i} things")
    roster.touch("Agent 7")
    shown = select_agents(roster.list_records(), "topic250 things")
    assert 10 <= len(shown) <= SHORTLIST
    names = [r.name for r in shown]
    assert "Agent 250" in names and "Agent 7" in names  # the match, and the most recently used


def test_archived_agents_are_never_shown(tmp_path):
    roster = AgentRoster(tmp_path / "r.json")
    with roster.batch():
        for i in range(40):
            roster.add_agent(f"Agent {i}", description=f"topic{i}")
    roster.archive_agent("Agent 5")
    assert "Agent 5" not in [r.name for r in select_agents(roster.list_records(), "topic5")]


def test_the_prompt_shows_descriptions_and_how_many_agents_are_hidden():
    roster = get_agent_roster()
    with roster.batch():
        for i in range(60):
            roster.add_agent(f"Agent {i}", description=f"Handles topic{i} for the user")
    text = _render_active_agents("what is the status of topic42")
    assert 'name="Agent 42"' in text and 'about="Handles topic42 for the user"' in text
    shown = text.count("<agent ")
    assert 10 <= shown <= SHORTLIST
    assert f'<other_agents count="{60 - shown}"' in text


def test_the_prompt_reaches_the_model_with_the_short_list():
    roster = get_agent_roster()
    with roster.batch():
        for i in range(200):
            roster.add_agent(f"Agent {i}", description=f"Handles topic{i}")
    content = prepare_message_with_history("update on topic77", "", "user")[0]["content"]
    assert 'name="Agent 77"' in content and 10 <= content.count("<agent ") <= SHORTLIST


def test_a_new_agents_description_and_people_come_from_its_first_request():
    assert extract_people("email Tess Gibson and cc bob@example.test about Friday") == ["bob@example.test", "Tess Gibson"]
    assert describe_request("x" * 500).endswith("…") and len(describe_request("x" * 500)) == 160


def test_the_right_agent_is_on_the_short_list_for_nearly_every_dataset_request(tmp_path):
    """Recall of the word ranker on the evaluator's 277 agents (the real-model eval relies on this)."""
    roster, truth = _dataset_roster(tmp_path)
    group_of = {a["name"]: a["group"] for a in truth["agents"]}
    records = roster.list_records()
    wanted = [r for r in truth["requests"] if r["expected_group"]]
    hits = sum(
        any(group_of[a.name] == r["expected_group"] for a in select_agents(records, r["text"]))
        for r in wanted
    )
    assert hits / len(wanted) >= 0.9, f"{hits}/{len(wanted)}"
