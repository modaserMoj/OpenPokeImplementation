"""Reminders (log 11): behaviour that works today and must keep working after the fixes."""

import asyncio
from datetime import datetime, timedelta, timezone

from helpers import run

from server.agents.execution_agent import batch_manager as bm
from server.services import trigger_scheduler
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore
from server.services.triggers.utils import to_storage_timestamp


def _service(tmp_path):
    return TriggerService(TriggerStore(tmp_path / "triggers.db"))


def _local(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def test_a_trigger_is_due_only_once_its_time_has_come(tmp_path):
    service = _service(tmp_path)
    now = datetime.now(timezone.utc)
    service.create_trigger(agent_name="Garbage", payload="take out the garbage",
                           start_time=_local(now + timedelta(minutes=2)), timezone_name="UTC")
    assert service.get_due_triggers(before=now) == []
    assert len(service.get_due_triggers(before=now + timedelta(minutes=3))) == 1


def test_local_start_time_is_stored_in_utc(tmp_path):
    service = _service(tmp_path)
    record = service.create_trigger(agent_name="Garbage", payload="x", start_time="2026-10-13T02:04:25",
                                    timezone_name="America/Toronto")
    assert record.next_trigger == "2026-10-13T06:04:25Z"  # Toronto is UTC-4 in October (what the live run stored)


def test_a_one_off_trigger_is_completed_after_it_fires(tmp_path):
    service = _service(tmp_path)
    record = service.create_trigger(agent_name="Garbage", payload="x",
                                    start_time=_local(datetime.now(timezone.utc)), timezone_name="UTC")
    updated = service.schedule_next_occurrence(record, fired_at=datetime.now(timezone.utc))
    assert updated.status == "completed" and updated.next_trigger is None


def test_a_recurring_trigger_is_rescheduled_for_the_next_day(tmp_path):
    service = _service(tmp_path)
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1)
    record = service.create_trigger(agent_name="Daily", payload="x", start_time=_local(start),
                                    recurrence_rule="FREQ=DAILY", timezone_name="UTC")
    fired_at = datetime.now(timezone.utc)
    updated = service.schedule_next_occurrence(record, fired_at=fired_at)
    assert updated.status == "active"
    next_fire = datetime.fromisoformat(updated.next_trigger.replace("Z", "+00:00"))
    assert timedelta(hours=1) < next_fire - fired_at <= timedelta(days=1, minutes=1)


def test_the_scheduler_runs_a_due_trigger_and_completes_it(tmp_path, monkeypatch, outbox):
    service = _service(tmp_path)
    ran = []

    async def fake_execute(self, instructions):
        ran.append((self.agent.name, instructions))
        return bm.ExecutionResult(agent_name=self.agent.name, success=True, response="reminder sent")

    monkeypatch.setattr(bm.ExecutionAgentRuntime, "execute", fake_execute)
    record = service.create_trigger(agent_name="Garbage", payload="take out the garbage",
                                    start_time=_local(datetime.now(timezone.utc) - timedelta(seconds=1)),
                                    timezone_name="UTC")

    async def scenario():
        scheduler = trigger_scheduler.TriggerScheduler()
        scheduler._service = service
        await scheduler._poll_once()
        await asyncio.sleep(0.05)
        await outbox.drain()

    run(scenario())
    assert ran and ran[0][0] == "Garbage" and "take out the garbage" in ran[0][1]
    assert outbox.delivered == ["[SUCCESS] Garbage: reminder sent"]
    assert service._store.fetch_one(record.id, "Garbage").status == "completed"



def test_a_brand_new_worker_is_told_the_current_time(llm):
    """Found live: a new agent had no clock (it used to read the time off its log) and refused to set
    "remind me in 3 minutes". Now every worker run and every main-agent turn says what time it is."""
    from helpers import completion

    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.utils.timezones import get_user_timezone_name

    llm.script(completion("ok"), completion(""))
    run(ExecutionAgentRuntime("Oven Reminder").execute("remind the user in 3 minutes to check the oven"))
    run(InteractionAgentRuntime().execute("hi"))
    worker, main = llm.calls
    assert "Current time for the user:" in worker["messages"][0]["content"]
    assert get_user_timezone_name() in worker["messages"][0]["content"]
    assert "<current_time>" in main["messages"][0]["content"]
