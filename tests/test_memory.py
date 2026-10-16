"""Issue 3: an agent's memory is bounded and compacted, but nothing is lost (log and full results stay on disk)."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from helpers import completion, run

from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.services.execution import get_execution_agent_logs, memory


def _runs(name, n, *, start=0, extra=""):
    logs = get_execution_agent_logs()
    for i in range(start, start + n):
        logs.record_request(name, f"task {i}: reply to thread_id=t{i} for alice{i}@example.test {extra}")
        logs.record_action(name, f"Calling gmail_reply_to_thread with: {{\"thread_id\": \"t{i}\"}}")
        logs.record_agent_response(name, f"done {i}: reply drafted, draft_id=d{i}. " + "detail " * 40)


class FakeSummariser:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def __call__(self, agent_name, previous, runs_text):
        self.calls.append((previous, runs_text))
        if self.fail:
            raise RuntimeError("summariser down")
        folded = runs_text.count("<agent_request")
        return (previous + " | " if previous else "") + f"Purpose: replies. Done: {folded} more runs."


def _log_lines(name):
    return list(get_execution_agent_logs().iter_entries(name))


# ------------------------------------------------------------------ runs and identifiers
def test_runs_are_grouped_even_when_a_request_was_never_logged():
    entries = [("agent_request", "", "a"), ("agent_response", "", "A"),
               ("agent_action", "", "old reminder run without a request"), ("agent_response", "", "B"),
               ("agent_request", "", "c")]
    assert [len(r) for r in memory.split_runs(entries)] == [2, 2, 1]


def test_identifiers_are_extracted_exactly():
    text = '{"draft_id": "r855", "threadId": "t-9"} sent to Bob@Example.test, id=m77'
    assert memory.extract_identifiers(text) == ["draft_id=r855", "threadId=t-9", "id=m77", "bob@example.test"]


def test_short_named_ids_are_kept_but_short_bare_ids_are_not():
    text = 'createTrigger succeeded | trigger_id=1, draft_id=d0, "id": 2'
    assert memory.extract_identifiers(text) == ["trigger_id=1", "draft_id=d0"]


# ------------------------------------------------------------------ the prompt view is bounded
def test_the_history_view_never_exceeds_the_budget_even_without_any_summary():
    _runs("Busy", 100)
    view = memory.render_history("Busy")
    assert len(view) <= memory.HISTORY_BUDGET_CHARS
    assert "task 99" in view  # the latest run is always there
    assert "older runs not shown" in view
    assert len(_log_lines("Busy")) == 300  # the archive is untouched


def test_a_single_huge_run_is_cut_from_the_front_not_dropped():
    get_execution_agent_logs().record_request("Huge", "x" * 20_000 + " THE END")
    view = memory.render_history("Huge")
    assert len(view) <= memory.HISTORY_BUDGET_CHARS and "THE END" in view and "omitted" in view


def test_a_short_history_is_shown_in_full():
    _runs("Fresh", 2)
    view = memory.render_history("Fresh")
    assert "task 0" in view and "task 1" in view and "Summary" not in view


# ------------------------------------------------------------------ compaction
def test_compaction_folds_older_runs_and_keeps_the_recent_ones_word_for_word():
    _runs("Mailer", 12)
    summariser = FakeSummariser()
    assert run(memory.compact("Mailer", summariser)) is True
    state = memory.load_state("Mailer")
    assert state.covered_runs == 12 - memory.RECENT_RUNS and "9 more runs" in state.summary
    view = memory.render_history("Mailer")
    assert "Summary of earlier work (9 earlier runs)" in view
    assert all(f"task {i}:" in view for i in (9, 10, 11)) and "task 8:" not in view
    assert "draft_id=d0" in view and "alice0@example.test" in view  # identifiers survive, extracted by code
    assert len(_log_lines("Mailer")) == 36  # nothing deleted


def test_compaction_does_nothing_while_the_history_is_small():
    _runs("Quiet", memory.RECENT_RUNS + 1)
    summariser = FakeSummariser()
    assert run(memory.compact("Quiet", summariser)) is False and summariser.calls == []


def test_compaction_is_incremental_the_old_summary_is_extended_not_rebuilt():
    _runs("Mailer", 12)
    summariser = FakeSummariser()
    run(memory.compact("Mailer", summariser))
    _runs("Mailer", 8, start=12)
    run(memory.compact("Mailer", summariser))
    previous, new_text = summariser.calls[1]
    assert previous.startswith("Purpose") and "task 0:" not in new_text and "task 9:" in new_text
    assert memory.load_state("Mailer").covered_runs == 20 - memory.RECENT_RUNS


def test_a_failed_summary_loses_nothing_and_the_view_stays_bounded():
    _runs("Mailer", 30)
    assert run(memory.compact("Mailer", FakeSummariser(fail=True))) is False
    assert memory.load_state("Mailer").covered_runs == 0
    assert len(memory.render_history("Mailer")) <= memory.HISTORY_BUDGET_CHARS
    assert run(memory.compact("Mailer", FakeSummariser())) is True  # and it simply works next time


def test_a_too_long_summary_is_capped():
    _runs("Mailer", 12)

    async def chatty(agent_name, previous, runs_text):
        return "word " * 5_000

    run(memory.compact("Mailer", chatty))
    assert len(memory.load_state("Mailer").summary) <= memory.SUMMARY_MAX_CHARS


def test_two_compactions_at_once_summarise_only_once():
    import asyncio

    _runs("Mailer", 12)
    summariser = FakeSummariser()

    async def both():
        return await asyncio.gather(memory.compact("Mailer", summariser), memory.compact("Mailer", summariser))

    assert sorted(run(both())) == [False, True] and len(summariser.calls) == 1


def test_a_finished_run_triggers_compaction_in_the_background(llm, monkeypatch):
    monkeypatch.setattr(memory, "AUTO_COMPACT", True)
    summariser = FakeSummariser()
    monkeypatch.setattr(memory, "summarise_runs", summariser)
    _runs("Mailer", 10)

    async def scenario():
        llm.script(completion("ok"))
        await ExecutionAgentRuntime("Mailer").execute("one more task")
        await memory.wait_for_compactions()

    run(scenario())
    assert len(summariser.calls) == 1 and memory.load_state("Mailer").covered_runs > 0


def test_the_summariser_call_uses_the_configured_model_and_a_capped_output(monkeypatch):
    seen = {}

    async def fake_request(**kwargs):
        seen.update(kwargs)
        return completion("Purpose: x. Done: y.")

    import server.openrouter_client as client_package
    monkeypatch.setattr(client_package, "request_chat_completion", fake_request)
    text = run(memory.summarise_runs("Mailer", "", "<agent_request>hi</agent_request>"))
    assert text.startswith("Purpose") and seen["max_tokens"] == 2000 and "Do not invent" in seen["system"]


def test_a_cut_off_summary_is_retried_with_more_room_and_never_saved(monkeypatch):
    """Found live: Haiku 5.5 spends output tokens reasoning, so a tight cap cut the summary after 'Purpose'."""
    budgets = []

    async def cut_off(**kwargs):
        budgets.append(kwargs["max_tokens"])
        response = completion("Purpose: replies.")
        response["choices"][0]["finish_reason"] = "length"
        return response

    import server.openrouter_client as client_package
    monkeypatch.setattr(client_package, "request_chat_completion", cut_off)
    _runs("Mailer", 12)
    assert run(memory.compact("Mailer")) is False  # nothing saved
    assert budgets == [2000, 4000] and memory.load_state("Mailer").summary == ""


# ------------------------------------------------------------------ tool results
def test_a_short_tool_result_is_logged_verbatim():
    assert ExecutionAgent("Tiny").record_tool_execution("listTriggers", "{}", '{"triggers": []}') == ""
    stored = [p for tag, _, p in get_execution_agent_logs().iter_entries("Tiny") if tag == "tool_response"]
    assert stored == ['listTriggers: {"triggers": []}']


def test_arguments_are_no_longer_cut_at_200_characters():
    args = json.dumps({"message_body": "y" * 900, "thread_id": "t-1"})
    ExecutionAgent("Args").record_tool_execution("gmail_reply_to_thread", args, "ok")
    action = [p for tag, _, p in get_execution_agent_logs().iter_entries("Args") if tag == "agent_action"][0]
    assert '"thread_id": "t-1"' in action


def test_an_unknown_or_malicious_result_id_is_refused():
    from server.agents.execution_agent.tools.memory_tools import build_registry
    get_tool_result = build_registry("Anyone")["get_tool_result"]
    assert "error" in get_tool_result("r_deadbeef")
    assert "error" in get_tool_result("../../../../etc/passwd")


def _store_results(name, count):
    for i in range(count):
        ExecutionAgent(name).record_tool_execution("gmail_search", "{}", json.dumps({"rows": ["z" * 50] * 20, "n": i}))


def test_old_full_results_are_pruned(monkeypatch):
    monkeypatch.setattr(memory, "KEEP_RESULTS", 5)
    monkeypatch.setattr(memory.ToolResultStore.prune, "__defaults__", (5,))
    _store_results("Searcher", 8)
    folder = get_execution_agent_logs()._base_dir / "results" / "searcher"
    assert len(list(folder.glob("r_*.json"))) == 5


def test_an_agent_with_an_active_reminder_keeps_every_full_result(monkeypatch, isolated_triggers):
    monkeypatch.setattr(memory.ToolResultStore.prune, "__defaults__", (5,))
    start = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    isolated_triggers.create_trigger(agent_name="Daily Digest", payload="send the digest", start_time=start,
                                     recurrence_rule="FREQ=DAILY", timezone_name="UTC")
    _store_results("Daily Digest", 8)
    folder = get_execution_agent_logs()._base_dir / "results" / "daily-digest"
    assert len(list(folder.glob("r_*.json"))) == 8


def test_ids_inside_search_results_do_not_push_out_the_ids_the_agent_acted_on():
    """Found by the memory eval: 40 runs of email searches evicted the draft/thread/reminder ids."""
    logs = get_execution_agent_logs()
    for i in range(30):
        logs.record_request("Searcher", f"run {i}")
        ExecutionAgent("Searcher").record_tool_execution(
            "task_email_search", "{}",
            json.dumps({"messages": [{"messageId": f"m{i}x{j}", "threadId": f"t{i}x{j}", "pad": "p" * 80} for j in range(10)]}))
        logs.record_agent_response("Searcher", "draft_id=KEEPME and trigger_id=3" if i == 1 else "nothing new")
    run(memory.compact("Searcher", FakeSummariser()))
    identifiers = memory.load_state("Searcher").identifiers
    assert "draft_id=KEEPME" in identifiers and "trigger_id=3" in identifiers
    assert not any(i.startswith("messageId=") for i in identifiers)
