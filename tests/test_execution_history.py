"""Issue 3 (log 03): a worker's whole log is pasted into its prompt, forever, and tool output is cut."""

import json

from helpers import completion, run

from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.services.execution import get_execution_agent_logs

MAX_PROMPT_CHARS_AFTER_50_RUNS = 20_000  # today: ~67,000 (about 23k tokens)


def test_earlier_runs_are_visible_to_the_next_run(llm):
    llm.script(completion("first answer: the invoice is #4411"), completion("second answer"))
    run(ExecutionAgentRuntime("Billing").execute("find the invoice"))
    run(ExecutionAgentRuntime("Billing").execute("what was the number?"))
    assert "the invoice is #4411" in llm.calls[1]["system"]


def test_worker_prompt_stays_bounded_after_many_runs(llm):
    for i in range(50):
        llm.script(completion(f"result {i}: " + "detail " * 55))
        get_execution_agent_logs().record_request("Recurring", f"daily task {i}")
        run(ExecutionAgentRuntime("Recurring").execute(f"daily task {i}"))
    system_prompt = ExecutionAgent("Recurring").build_system_prompt_with_history()
    assert len(system_prompt) <= MAX_PROMPT_CHARS_AFTER_50_RUNS, len(system_prompt)


def test_a_long_tool_result_is_never_silently_cut_the_full_result_stays_retrievable():
    from server.agents.execution_agent.tools.memory_tools import build_registry

    big = json.dumps({"messages": [{"id": f"m{i}", "subject": f"Subject {i}"} for i in range(400)]})
    result_id = ExecutionAgent("Searcher").record_tool_execution("gmail_list_drafts", "{}", big)
    stored = [p for tag, _, p in get_execution_agent_logs().iter_entries("Searcher") if tag == "tool_response"][0]
    assert "summary only" in stored and result_id in stored and len(stored) < 1000  # marked, short, points at the full one
    assert "id=m0" in stored and "Subject 0" in stored  # the digest keeps what identifies the items
    fetched = build_registry("Searcher")["get_tool_result"](result_id)
    assert fetched["result"] == big


def test_the_instruction_reaches_the_model_once(llm, monkeypatch):
    """Through the real path: the main agent's tool hands the work to a worker run."""
    import asyncio

    from server.agents.interaction_agent import tools as interaction_tools
    instruction = "UNIQUE-INSTRUCTION-42 draft a note to Sam"

    async def scenario():
        interaction_tools.send_message_to_agent("Email to Sam", instruction)
        for _ in range(100):
            if llm.calls:
                break
            await asyncio.sleep(0.01)

    run(scenario())
    call = llm.calls[0]
    seen = call["system"].count(instruction) + sum(m["content"].count(instruction) for m in call["messages"])
    assert seen == 1, seen
    requests = [p for tag, _, p in get_execution_agent_logs().iter_entries("Email to Sam") if tag == "agent_request"]
    assert requests == [instruction]  # still recorded exactly once, by the run


def test_a_reminder_run_records_its_request_too(llm):
    llm.script(completion("reminded"))
    run(ExecutionAgentRuntime("Garbage").execute("Trigger fired: take out the garbage"))
    tags = [tag for tag, _, _ in get_execution_agent_logs().iter_entries("Garbage")]
    assert tags == ["agent_request", "agent_response"]
