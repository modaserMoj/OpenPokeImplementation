"""Issues 4-5: background and parallel results reach the main agent through the outbox.

(The outbox's own timing rules are tested in test_outbox.py.)"""

import asyncio
import time

from helpers import completion, run

from server.agents.execution_agent import batch_manager as bm
from server.services import trigger_scheduler
from server.services.outbox import REMINDER
from server.services.triggers.models import TriggerRecord


def _fake_runs(monkeypatch, durations):
    async def fake_execute(self, instructions):
        await asyncio.sleep(durations.get(self.agent.name, 0.01))
        return bm.ExecutionResult(agent_name=self.agent.name, success=True, response=f"{self.agent.name} done")

    monkeypatch.setattr(bm.ExecutionAgentRuntime, "execute", fake_execute)


def test_a_finished_result_is_delivered_without_waiting_for_a_slow_agent(monkeypatch, outbox):
    _fake_runs(monkeypatch, {"fast agent": 0.01, "slow agent": 0.6})
    stamps = []

    async def capture(payload):
        stamps.append((time.monotonic(), payload))

    outbox._deliver = capture

    async def scenario():
        manager = bm.ExecutionBatchManager()
        started = time.monotonic()
        task = asyncio.gather(manager.execute_agent("fast agent", "x"), manager.execute_agent("slow agent", "x"))
        await asyncio.sleep(0.3)  # the fast agent is long finished, the slow one is not
        early = [payload for at, payload in stamps if at - started < 0.3]
        await task
        await outbox.drain()
        return early

    early = run(scenario())
    assert any("fast agent done" in p for p in early), "the fast result was held back"
    assert any("slow agent done" in p for _, p in stamps)  # and the slow one still arrives


def test_two_requests_to_the_same_agent_do_not_overlap(monkeypatch, llm):
    async def slow_answer(kwargs):
        await asyncio.sleep(0.05)
        return completion("ANSWER-" + kwargs["messages"][0]["content"])

    llm.script(slow_answer, slow_answer)

    async def scenario():
        manager = bm.ExecutionBatchManager()
        await asyncio.gather(manager.execute_agent("Email to Alice", "task-A"),
                             manager.execute_agent("email to alice", "task-B"))  # same agent, other spelling

    run(scenario())
    later_prompt = llm.calls[1]["system"]
    assert "ANSWER-task-A" in later_prompt or "ANSWER-task-B" in later_prompt


def test_different_agents_still_run_in_parallel(monkeypatch):
    _fake_runs(monkeypatch, {"A": 0.2, "B": 0.2, "C": 0.2})

    async def scenario():
        manager = bm.ExecutionBatchManager()
        started = time.monotonic()
        await asyncio.gather(*(manager.execute_agent(n, "x") for n in "ABC"))
        return time.monotonic() - started

    assert run(scenario()) < 0.45  # about 0.2s, not 0.6s


def test_reminders_due_together_reach_the_main_agent_as_one_turn(monkeypatch, outbox):
    _fake_runs(monkeypatch, {})

    class StubService:
        def schedule_next_occurrence(self, *a, **k): pass
        def record_failure(self, *a, **k): pass
        def clear_next_fire(self, *a, **k): pass

    stamp = "2026-10-13T09:00:00Z"
    due = [TriggerRecord(id=i, agent_name=f"Reminder {i}", payload="ping", status="active",
                         next_trigger=stamp, created_at=stamp, updated_at=stamp) for i in range(1, 6)]

    async def scenario():
        scheduler = trigger_scheduler.TriggerScheduler()
        scheduler._service = StubService()
        await asyncio.gather(*(scheduler._execute_trigger(t) for t in due))
        await outbox.drain()

    run(scenario())
    assert len(outbox.delivered) == 1, f"{len(outbox.delivered)} separate turns for 5 reminders"
    assert all(f"Reminder {i} done" in outbox.delivered[0] for i in range(1, 6))


def test_reminder_runs_are_reported_as_reminders(monkeypatch, outbox):
    _fake_runs(monkeypatch, {})
    outbox._auto_schedule = False

    async def scenario():
        await bm.ExecutionBatchManager(kind=REMINDER).execute_agent("Gym", "ping")

    run(scenario())
    assert [item.kind for item in outbox.pending()] == ["reminder"]


def test_a_failed_run_is_still_reported(monkeypatch, outbox):
    async def broken(self, instructions):
        raise RuntimeError("model down")

    monkeypatch.setattr(bm.ExecutionAgentRuntime, "execute", broken)

    async def scenario():
        result = await bm.ExecutionBatchManager().execute_agent("Mailer", "x")
        await outbox.drain()
        return result

    assert run(scenario()).success is False
    assert outbox.delivered and outbox.delivered[0].startswith("[FAILED] Mailer")
