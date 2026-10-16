"""Issue 9: the main agent's chat history in the prompt stays under a budget (about 4,000 tokens).

Summarising starts when the budget is passed (not only after 110 messages), the summary itself is capped, and the
prompt view is bounded even when summarising has not run or failed. The conversation log itself is never cut."""

import pytest
from helpers import completion, run, tool_call

from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.config import get_settings
from server.services.conversation import get_conversation_log, get_working_memory_log
from server.services.conversation.summarization import summarizer as summarizer_module
from server.openrouter_client import OpenRouterError

BUDGET = get_settings().conversation_history_budget_chars


def _chat(n, size=300, start=0):
    log = get_conversation_log()
    for i in range(start, start + n):
        text = f"message {i}: " + "x" * size
        (log.record_user_message if i % 3 == 0 else log.record_reply if i % 3 == 1 else log.record_agent_message)(text)


class FakeSummariser:
    def __init__(self, text="Summary generated: 2026-10-14\nPending & Follow-ups:\n- reply to Dana", fail=False):
        self.calls, self.text, self.fail = [], text, fail

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise OpenRouterError("summariser down")
        return completion(self.text)


def test_the_budget_is_about_4000_tokens():
    assert get_settings().conversation_history_budget_tokens == 4000 and BUDGET == 14_000


def test_the_prompt_view_never_exceeds_the_budget_even_before_any_summary():
    _chat(200)
    view = get_working_memory_log().render_transcript(budget_chars=BUDGET)
    assert len(view) <= BUDGET
    assert "message 199:" in view and "older messages are not shown" in view
    unbounded = get_working_memory_log().render_transcript()
    assert len(unbounded) > 4 * BUDGET  # without a budget the old behaviour is unchanged
    assert len(list(get_conversation_log().iter_entries())) == 200  # the log itself is never cut


def test_a_single_huge_message_is_cut_from_the_front():
    get_conversation_log().record_agent_message("y" * 50_000 + " LATEST PART")
    view = get_working_memory_log().render_transcript(budget_chars=BUDGET)
    assert len(view) <= BUDGET and view.endswith("LATEST PART</agent_message>")


def test_the_main_agent_prompt_uses_the_bounded_history(llm):
    _chat(200)
    llm.script(completion("", [tool_call("send_message_to_user", {"message": "hi"})]))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] X: done"))
    content = llm.calls[0]["messages"][0]["content"]
    history = content.split("<conversation_history>")[1].split("</conversation_history>")[0]
    assert len(history) <= BUDGET + 2 and "message 199:" in history


def test_summarising_starts_when_the_history_passes_the_budget_not_after_110_messages(monkeypatch):
    fake = FakeSummariser()
    monkeypatch.setattr(summarizer_module, "request_chat_completion", fake)
    _chat(40, size=500)  # ~22k characters: over budget, far below 110 messages
    assert run(summarizer_module.summarize_conversation()) is True
    state = get_working_memory_log().load_summary_state()
    assert state.summary_text.startswith("Summary generated")
    assert len(state.unsummarized_entries) == get_settings().conversation_summary_tail_size  # last 10 kept verbatim
    view = get_working_memory_log().render_transcript(budget_chars=BUDGET)
    assert "<conversation_summary>" in view and "message 39:" in view and "message 29:" not in view


def test_a_short_history_is_not_summarised(monkeypatch):
    fake = FakeSummariser()
    monkeypatch.setattr(summarizer_module, "request_chat_completion", fake)
    _chat(20, size=100)
    assert run(summarizer_module.summarize_conversation()) is False and fake.calls == []


def test_the_message_count_rule_still_works_as_a_backstop(monkeypatch):
    fake = FakeSummariser()
    monkeypatch.setattr(summarizer_module, "request_chat_completion", fake)
    _chat(110, size=10)  # tiny messages: under budget, but 110 of them
    assert run(summarizer_module.summarize_conversation()) is True


def test_the_summary_is_asked_to_stay_short_and_is_capped(monkeypatch):
    fake = FakeSummariser(text="word " * 5_000)
    monkeypatch.setattr(summarizer_module, "request_chat_completion", fake)
    _chat(40, size=500)
    run(summarizer_module.summarize_conversation())
    assert fake.calls[0]["max_tokens"] == summarizer_module.SUMMARY_MAX_TOKENS == 2000
    assert "under about 500 words" in fake.calls[0]["system"]
    assert len(get_working_memory_log().load_summary_state().summary_text) <= summarizer_module.SUMMARY_MAX_CHARS


def test_a_failed_summary_changes_nothing_and_the_view_stays_bounded(monkeypatch):
    monkeypatch.setattr(summarizer_module, "request_chat_completion", FakeSummariser(fail=True))
    _chat(60, size=500)
    with pytest.raises(OpenRouterError):
        run(summarizer_module.summarize_conversation())
    state = get_working_memory_log().load_summary_state()
    assert state.summary_text == "" and state.last_index == -1
    assert len(get_working_memory_log().render_transcript(budget_chars=BUDGET)) <= BUDGET


def test_after_summarising_the_history_shrinks_well_below_the_budget(monkeypatch):
    monkeypatch.setattr(summarizer_module, "request_chat_completion", FakeSummariser())
    _chat(40, size=500)
    before = len(get_working_memory_log().render_transcript())
    run(summarizer_module.summarize_conversation())
    after = len(get_working_memory_log().render_transcript(budget_chars=BUDGET))
    assert after < before / 3
