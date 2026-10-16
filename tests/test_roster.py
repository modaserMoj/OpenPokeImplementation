"""Issues 1-2 and S3 (logs 01, 02, 07, 08): the agent list is a plain list of names."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from server.services.execution import get_agent_roster, get_execution_agent_logs
from server.services.execution import roster as roster_module

ROOT = Path(__file__).resolve().parent.parent


def test_added_agents_survive_a_reload():
    roster = get_agent_roster()
    roster.add_agent("Email to Alice")
    roster.add_agent("Email to Bob")
    reloaded = roster_module.AgentRoster(roster._roster_path)
    assert reloaded.get_agents() == ["Email to Alice", "Email to Bob"]


def test_adding_the_same_name_twice_keeps_one_entry():
    roster = get_agent_roster()
    roster.add_agent("Email to Alice")
    roster.add_agent("Email to Alice")
    assert roster.get_agents() == ["Email to Alice"]


def test_a_busy_lock_or_failed_save_does_not_empty_the_roster_file():
    fcntl = pytest.importorskip("fcntl")
    roster = get_agent_roster()
    roster.add_agent("Email to Alice")
    roster.add_agent("Email to Bob")
    holder = open(roster._roster_path, "a")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    try:
        roster.add_agent("Email to Carol")
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
    saved = json.loads(roster._roster_path.read_text())
    assert [a["name"] for a in saved["agents"]][:2] == ["Email to Alice", "Email to Bob"]


def test_roster_module_imports_without_fcntl():
    code = (
        "import sys, os; sys.modules['fcntl'] = None\n"
        "os.environ['OPENROUTER_API_KEY'] = 'x'\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import server.services.execution.roster\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-300:]


def test_agents_in_the_roster_do_not_share_a_log_file():
    roster = get_agent_roster()
    for name in ("Email to Alice", "email to alice", "Email to Alice!"):
        roster.add_agent(name)
    logs = get_execution_agent_logs()
    paths = {logs._log_path(name) for name in roster.get_agents()}
    assert len(paths) == len(roster.get_agents())


def test_an_agent_can_be_retired_without_wiping_the_others():
    roster = get_agent_roster()
    retire = next((getattr(roster, n) for n in ("archive_agent", "remove_agent", "retire_agent") if hasattr(roster, n)), None)
    assert retire is not None, "no archive/remove/retire method on the roster"
    roster.add_agent("Old thread")
    roster.add_agent("Live thread")
    retire("Old thread")
    assert roster.get_agents() == ["Live thread"]


def test_prompt_size_does_not_grow_with_the_number_of_agents():
    from server.agents.interaction_agent.agent import prepare_message_with_history

    roster = get_agent_roster()
    sizes = {}
    for count in (10, 1000):
        roster.clear()
        with roster.batch():
            for i in range(count):
                roster.add_agent(f"Email to Person{i} about topic {i}")
        sizes[count] = len(prepare_message_with_history("hi", "", "user")[0]["content"])
    assert sizes[1000] < 3 * sizes[10], sizes
