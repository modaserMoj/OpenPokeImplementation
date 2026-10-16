"""Issue 6: each LLM role has its own, configurable model; Sonnet only for the main agent and the worker."""

import pytest

from server.config import HAIKU, SONNET, Settings

ROLES = {
    "interaction_agent_model": ("OPENPOKE_MODEL_INTERACTION", SONNET),
    "execution_agent_model": ("OPENPOKE_MODEL_EXECUTION", SONNET),
    "execution_agent_search_model": ("OPENPOKE_MODEL_EMAIL_SEARCH", HAIKU),
    "summarizer_model": ("OPENPOKE_MODEL_SUMMARIZER", HAIKU),
    "email_classifier_model": ("OPENPOKE_MODEL_EMAIL_CLASSIFIER", HAIKU),
}


def test_defaults_sonnet_for_judgment_haiku_for_simple_jobs(monkeypatch):
    for env, _ in ROLES.values():
        monkeypatch.delenv(env, raising=False)
    settings = Settings()
    assert {field: getattr(settings, field) for field in ROLES} == {field: model for field, (_, model) in ROLES.items()}
    assert SONNET == "anthropic/claude-sonnet-5.5" and HAIKU == "anthropic/claude-haiku-5.5"


@pytest.mark.parametrize("field", list(ROLES))
def test_every_role_can_be_overridden_from_the_environment(monkeypatch, field):
    env, _ = ROLES[field]
    monkeypatch.setenv(env, "some/other-model")
    assert getattr(Settings(), field) == "some/other-model"


def test_each_call_site_reads_its_own_role(monkeypatch, llm, gmail):
    """The main agent and worker send their configured model to the API."""
    from helpers import completion, run

    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime
    from server import config

    custom = Settings().model_copy(update={"interaction_agent_model": "test/main", "execution_agent_model": "test/worker"})
    monkeypatch.setattr(config, "get_settings", lambda: custom)
    import server.agents.execution_agent.runtime as er
    import server.agents.interaction_agent.runtime as ir
    monkeypatch.setattr(er, "get_settings", lambda: custom)
    monkeypatch.setattr(ir, "get_settings", lambda: custom)
    llm.script(completion("ok"), completion(""))
    run(ExecutionAgentRuntime("Worker").execute("x"))
    run(InteractionAgentRuntime().execute("hi"))
    assert [c["model"] for c in llm.calls] == ["test/worker", "test/main"]
