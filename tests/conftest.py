"""Shared fixtures: no network, throw-away data, a scripted fake model and a fake Gmail layer.

Nothing in this suite calls OpenRouter, Composio or Gmail, and nothing touches server/data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must happen before the server package is imported: fake credentials, and empty values so that
# server/config.py does not pick up the developer's real keys from .env.
os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
os.environ["OPENPOKE_EMBEDDING_MODEL"] = ""  # tests never call the embedding service
os.environ["COMPOSIO_API_KEY"] = ""
os.environ["COMPOSIO_GMAIL_AUTH_CONFIG_ID"] = ""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

from helpers import completion  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any code path tries to reach a real service."""

    async def _blocked(self, *args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(httpx.AsyncClient, "post", _blocked)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point every file-backed singleton at a temp dir so tests never see (or touch) live data."""
    from server.services.conversation import get_working_memory_log
    from server.services.conversation import log as conversation_log
    from server.services.execution import log_store, roster

    monkeypatch.setattr(log_store, "_execution_agent_logs", log_store.ExecutionAgentLogStore(tmp_path / "agents"))
    monkeypatch.setattr(roster, "_agent_roster", roster.AgentRoster(tmp_path / "roster.json"))

    clog = conversation_log.get_conversation_log()
    working = get_working_memory_log()
    monkeypatch.setattr(clog, "_path", tmp_path / "conversation.log")
    monkeypatch.setattr(working, "_path", tmp_path / "working_memory.log")
    working._initialize_file()
    monkeypatch.setattr(type(clog), "_notify_summarization", lambda self: None)
    return tmp_path


class FakeLLM:
    """Scripted stand-in for request_chat_completion.

    script(...) takes responses in order. An item may be a dict (returned as is), an Exception
    (raised) or a callable(kwargs) -> dict | awaitable. When the script runs out it answers with an
    empty message, which ends an agent loop. Every call is kept in .calls for assertions.
    """

    def __init__(self):
        self._queue = []
        self.calls = []

    def script(self, *items):
        self._queue.extend(items)
        return self

    async def __call__(self, **kwargs):
        system = kwargs.get("system")
        if isinstance(system, (list, tuple)):  # the worker sends [base prompt, memory] as separate cache blocks
            kwargs = {**kwargs, "system_blocks": list(system), "system": "\n\n".join(system)}
        self.calls.append(kwargs)
        if not self._queue:
            return completion("")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(kwargs)
            if hasattr(item, "__await__"):
                item = await item
        return item


@pytest.fixture
def llm(monkeypatch):
    """Fake model patched into BOTH the interaction agent and the execution agent."""
    from server.agents.execution_agent import runtime as execution_runtime
    from server.agents.interaction_agent import runtime as interaction_runtime

    fake = FakeLLM()
    monkeypatch.setattr(execution_runtime, "request_chat_completion", fake)
    monkeypatch.setattr(interaction_runtime, "request_chat_completion", fake)
    return fake


class FakeGmail:
    def __init__(self):
        self.calls = []  # (composio tool name, arguments)
        self._drafts = 0

    @property
    def sent(self):
        send_tools = {"GMAIL_SEND_DRAFT", "GMAIL_REPLY_TO_THREAD", "GMAIL_FORWARD_MESSAGE"}
        return [c for c in self.calls if c[0] in send_tools]


@pytest.fixture
def gmail(monkeypatch):
    """Fake Gmail layer: records every Composio call, returns success, never leaves the process."""
    from server.agents.execution_agent.tools import gmail as gmail_tools

    fake = FakeGmail()

    def _execute(tool_name, composio_user_id, arguments):
        fake.calls.append((tool_name, arguments))
        # The exact shape Composio returns on success, including "error": None (found live: code that checked
        # `"error" not in result` treated every successful draft as a failure, and a fake without the key hid it).
        data = {"id": "fake-id"}
        if tool_name == "GMAIL_CREATE_EMAIL_DRAFT":
            fake._drafts += 1
            data = {"id": f"draft-{fake._drafts}", "display_url": "https://mail.google.com/mail/u/0/#drafts/x",
                    "message": {"id": "1a13d8c765244e0d", "labelIds": ["DRAFT"], "threadId": "1a13d8c765244e0d"}}
        return {"data": data, "error": None, "log_id": "log_test", "session_info": None, "successful": True}

    monkeypatch.setattr(gmail_tools, "_execute", _execute)
    monkeypatch.setattr(gmail_tools, "get_active_gmail_user_id", lambda: "test-user")
    return fake


@pytest.fixture(autouse=True)
def fresh_approvals(monkeypatch):
    """Every test starts with no pending or granted send approvals."""
    from server.services import approvals

    monkeypatch.setattr(approvals, "_store", approvals.ApprovalStore())
    return approvals.get_approval_store()


@pytest.fixture(autouse=True)
def isolated_triggers(tmp_path, monkeypatch):
    """Reminders live in a temp database (memory retention asks which agents have an active reminder)."""
    from server.services import triggers
    from server.services.triggers.service import TriggerService
    from server.services.triggers.store import TriggerStore

    service = TriggerService(TriggerStore(tmp_path / "triggers.db"))
    monkeypatch.setattr(triggers, "_trigger_service", service)
    return service


@pytest.fixture(autouse=True)
def no_background_compaction(monkeypatch):
    """Tests call memory.compact() themselves, so nothing runs after a test's event loop has closed."""
    from server.services.execution import memory

    monkeypatch.setattr(memory, "AUTO_COMPACT", False)


FAST_WINDOWS = {"requested": 0.02, "reminder": 0.05, "email": 0.1}
FAST_EMAIL_COOLDOWNS = (0.2, 0.3, 0.5, 0.8, 1.3, 2.1)  # the real 2,3,5,8,13,21 minutes, scaled down


@pytest.fixture(autouse=True)
def outbox(monkeypatch):
    """A fast outbox whose deliveries are captured in .delivered (nothing reaches a real main agent)."""
    from server.services import outbox as outbox_module

    delivered = []

    async def capture(payload):
        delivered.append(payload)

    box = outbox_module.Outbox(capture, windows=FAST_WINDOWS, email_cooldowns=FAST_EMAIL_COOLDOWNS, reset_after=6.0)
    box.delivered = delivered
    monkeypatch.setattr(outbox_module, "_outbox", box)
    return box


@pytest.fixture(autouse=True)
def isolated_gmail_account(tmp_path, monkeypatch):
    """The connected-Gmail-account file lives in a temp dir (it is persisted to disk in production)."""
    from server.services.gmail import client as gmail_client

    monkeypatch.setattr(gmail_client, "_ACTIVE_USER_PATH", tmp_path / "gmail_active_user.json")
    monkeypatch.setattr(gmail_client, "_ACTIVE_USER_ID", None)
    monkeypatch.setattr(gmail_client, "_ACTIVE_USER_LOADED", False)
