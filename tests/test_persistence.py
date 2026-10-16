"""Everything that matters survives a restart, and "Clear history" clears the chat - not the agents."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from helpers import run

from server.services import approvals as approvals_module
from server.services.approvals import ApprovalStore, begin_turn, end_turn
from server.services.gmail import client as gmail_client
from server.services.outbox import EMAIL, REMINDER, Outbox


def _restart_gmail_module(monkeypatch):
    monkeypatch.setattr(gmail_client, "_ACTIVE_USER_ID", None)
    monkeypatch.setattr(gmail_client, "_ACTIVE_USER_LOADED", False)


# ------------------------------------------------------------------ the connected Gmail account
def test_the_connected_gmail_account_survives_a_restart(monkeypatch):
    gmail_client._set_active_gmail_user_id("web-123")
    _restart_gmail_module(monkeypatch)
    assert gmail_client.get_active_gmail_user_id() == "web-123"  # before: None, Gmail silently disconnected


def test_disconnecting_is_remembered_too(monkeypatch):
    gmail_client._set_active_gmail_user_id("web-123")
    gmail_client._set_active_gmail_user_id(None)
    _restart_gmail_module(monkeypatch)
    assert gmail_client.get_active_gmail_user_id() is None


# ------------------------------------------------------------------ send approvals
def test_a_draft_shown_before_a_restart_can_be_approved_and_sent_after_it(tmp_path):
    path = tmp_path / "approvals.json"
    before = ApprovalStore(path)
    before.propose("draft", recipients=["a@example.test"], subject="Hi", body="Hello there", draft_id="d-1")
    before.mark_shown("a@example.test", "Hello there")

    after = ApprovalStore(path)  # a new process
    token = begin_turn("user", "yes send it")
    try:
        proposal, reason = after.approve()
    finally:
        end_turn(token)
    assert proposal is not None, reason
    assert after.consume_draft("d-1") is True
    assert ApprovalStore(path).consume_draft("d-1") is False  # used approvals are not brought back


def test_expired_proposals_are_not_brought_back(tmp_path, monkeypatch):
    path = tmp_path / "approvals.json"
    now = [1_000_000.0]
    monkeypatch.setattr(approvals_module, "_now", lambda: now[0])
    ApprovalStore(path).propose("draft", recipients=["a@example.test"], body="x", draft_id="d-1")
    now[0] += approvals_module.APPROVAL_TTL_SECONDS + 1
    assert ApprovalStore(path).candidates() == [] and ApprovalStore(path)._proposals == {}


# ------------------------------------------------------------------ the outbox
def test_undelivered_outbox_items_are_delivered_after_a_restart(tmp_path):
    path = tmp_path / "outbox.json"
    before = Outbox(state_path=path, auto_schedule=False)
    before.add(REMINDER, "[SUCCESS] Pills: take pills")
    before.add(EMAIL, "Important email watcher notification:\nInvoice due")

    delivered = []

    async def capture(payload):
        delivered.append(payload)

    after = Outbox(capture, state_path=path, auto_schedule=False)
    assert after.restore() == 2

    run(after.flush_now())
    assert len(delivered) == 1 and "take pills" in delivered[0] and "Invoice due" in delivered[0]
    assert json.loads(path.read_text()) == []  # delivered, so no longer on disk


def test_an_item_stays_on_disk_until_its_delivery_has_finished(tmp_path):
    path = tmp_path / "outbox.json"
    gate = asyncio.Event()

    async def slow_delivery(payload):
        await gate.wait()

    box = Outbox(slow_delivery, state_path=path, auto_schedule=False)
    box.add(REMINDER, "stand-up at 9")

    async def scenario():
        flushing = asyncio.create_task(box.flush_now())
        await asyncio.sleep(0.02)
        during = json.loads(path.read_text())   # a crash right now would not lose it
        gate.set()
        await flushing
        return during

    during = run(scenario())
    assert [i["text"] for i in during] == ["stand-up at 9"] and json.loads(path.read_text()) == []


# ------------------------------------------------------------------ clearing the chat
def _setup_agent_with_everything(isolated_triggers):
    from server.agents.execution_agent.agent import ExecutionAgent
    from server.services.conversation import get_conversation_log
    from server.services.execution import get_agent_roster, memory

    get_conversation_log().record_user_message("hi")
    get_agent_roster().add_agent("Daily digest")
    memory.save_state("Daily digest", memory.MemoryState(summary="sends the digest at 8"))
    ExecutionAgent("Daily digest").record_tool_execution("search", "{}", json.dumps({"rows": ["x" * 80] * 20}))
    start = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    isolated_triggers.create_trigger(agent_name="Daily digest", payload="digest", start_time=start,
                                     recurrence_rule="FREQ=DAILY", timezone_name="UTC")


def test_clear_history_clears_the_chat_but_keeps_agents_memory_and_reminders(isolated_triggers):
    from server.routes.chat import clear_history
    from server.services.conversation import get_conversation_log
    from server.services.execution import get_agent_roster, memory

    _setup_agent_with_everything(isolated_triggers)
    clear_history()
    assert list(get_conversation_log().iter_entries()) == []
    assert get_agent_roster().get_agents() == ["Daily digest"]
    assert memory.load_state("Daily digest").summary == "sends the digest at 8"
    assert len(isolated_triggers.list_triggers(agent_name="Daily digest")) == 1


def test_a_full_reset_is_explicit_and_clears_every_store(isolated_triggers):
    from server.routes.chat import clear_history
    from server.services.execution import get_agent_roster, get_execution_agent_logs, memory

    _setup_agent_with_everything(isolated_triggers)
    clear_history(everything=True)
    base = get_execution_agent_logs()._base_dir
    assert get_agent_roster().get_agents() == []
    assert isolated_triggers.list_triggers(agent_name="Daily digest") == []
    assert memory.load_state("Daily digest").summary == ""
    assert not (base / "results").exists() and not list(base.glob("*.memory.json"))



def test_clearing_the_chat_drops_drafts_waiting_for_approval(llm, gmail, isolated_triggers):
    """Found in live test 7: a draft shown in the cleared chat stayed approvable, so a later "yes" could have sent
    an email the user could no longer see."""
    from helpers import completion, tool_call

    from server.agents.interaction_agent import tools as interaction_tools
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server.routes.chat import clear_history
    from server.services.approvals import get_approval_store

    interaction_tools.send_draft(to="a@example.test", subject="Dinner", body="7 PM works for dinner.")
    assert get_approval_store().candidates()
    clear_history()
    assert get_approval_store().candidates() == []
    llm.script(completion("", [tool_call("send_message_to_user", {"message": "Yes to what?"})]))
    run(InteractionAgentRuntime().execute("yes"))
    assert "<approval_note>" not in llm.calls[0]["messages"][0]["content"]
