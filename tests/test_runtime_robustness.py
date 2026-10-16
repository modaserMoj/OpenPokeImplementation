"""Issues 7-8 and the 'other findings' (logs 09, 10, 11): how the two agent loops handle bad input,
failures and repeats."""

import json

from helpers import completion, run, tool_call

from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.agents.interaction_agent import runtime as interaction_runtime
from server.agents.interaction_agent import tools as interaction_tools
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import TOOL_SCHEMAS
from server.openrouter_client import OpenRouterError
from server.services.conversation import get_conversation_log

BROKEN_JSON = '{"thread_id": "t1", "recipient_email": "a@b.test", "message_body": "Hel'


def _tool_message_text(call):
    return " ".join(m["content"] for m in call["messages"] if m.get("role") == "tool").lower()


def _replies():
    return [p for tag, _, p in get_conversation_log().iter_entries() if tag == "poke_reply"]


def test_main_agent_tells_the_model_when_its_tool_json_is_invalid(llm):
    llm.script(completion("", [tool_call("send_message_to_user", '{"message": "hi')]), completion(""))
    run(InteractionAgentRuntime().execute("hello"))
    assert "invalid json" in _tool_message_text(llm.calls[1])


def test_worker_tells_the_model_when_its_tool_json_is_invalid(llm, gmail):
    llm.script(completion("", [tool_call("gmail_reply_to_thread", BROKEN_JSON)]), completion("done"))
    run(ExecutionAgentRuntime("Mailer").execute("reply to the thread"))
    assert "json" in _tool_message_text(llm.calls[1])


def test_prompt_only_mentions_tools_that_exist():
    prompt = interaction_runtime.build_system_prompt()
    defined = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "reacttomessage" in defined or "reacttomessage" not in prompt


def test_relaying_a_result_to_the_user_takes_one_model_call(llm):
    """Reminder firing, worker results: the model messages the user and there is nothing left to decide."""
    llm.script(completion("", [tool_call("send_message_to_user", {"message": "Time to take out the garbage."})]))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] Garbage Reminder: it is time"))
    assert len(llm.calls) == 1, len(llm.calls)


def test_a_turn_that_delegated_and_then_told_the_user_ends_without_an_empty_extra_call(llm, monkeypatch):
    sent = []
    monkeypatch.setattr(interaction_tools, "send_message_to_agent",
                        lambda agent_name, instructions: (sent.append(agent_name), interaction_tools.ToolResult(True, {"status": "submitted"}))[1])
    llm.script(completion("", [tool_call("send_message_to_agent", {"agent_name": "Garbage", "instructions": "remind me"})]),
               completion("", [tool_call("send_message_to_user", {"message": "On it."})]))
    run(InteractionAgentRuntime().execute("remind me in 2 minutes to take out the garbage"))
    assert sent == ["Garbage"] and len(llm.calls) == 2  # was 3: the third call returned nothing


def test_an_acknowledgement_alone_does_not_end_the_turn_so_the_delegation_still_happens(llm, monkeypatch):
    """The model sometimes acknowledges first and delegates in its next call; the turn must stay open."""
    sent = []
    monkeypatch.setattr(interaction_tools, "send_message_to_agent",
                        lambda agent_name, instructions: (sent.append(agent_name), interaction_tools.ToolResult(True, {"status": "submitted"}))[1])
    llm.script(completion("", [tool_call("send_message_to_user", {"message": "Let me check your email."})]),
               completion("", [tool_call("send_message_to_agent", {"agent_name": "Email Checker", "instructions": "latest email"})]))
    run(InteractionAgentRuntime().execute("who sent my latest email?"))
    assert sent == ["Email Checker"]


def test_the_turn_stays_open_when_a_tool_failed_so_the_model_can_react(llm):
    llm.script(completion("", [tool_call("send_message_to_user", '{"message": "oops')]),  # invalid JSON
               completion("", [tool_call("send_message_to_user", {"message": "Sorry, one more try."})]))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] Agent: done"))
    assert len(llm.calls) == 2 and "Sorry, one more try." in _replies()


def test_the_turn_stays_open_after_an_approval_because_the_send_comes_next(llm, gmail, monkeypatch):
    sent = []
    monkeypatch.setattr(interaction_tools, "send_message_to_agent",
                        lambda agent_name, instructions: (sent.append(agent_name), interaction_tools.ToolResult(True, {"status": "submitted"}))[1])
    from server.agents.execution_agent.tools import gmail as gmail_tools
    gmail_tools.gmail_create_draft(recipient_email="a@example.test", subject="Hi", body="Hello there")
    interaction_tools.send_draft(to="a@example.test", subject="Hi", body="Hello there")
    llm.script(completion("", [tool_call("approve_send", {})]),
               completion("", [tool_call("send_message_to_agent", {"agent_name": "Mailer", "instructions": "send it"})]))
    run(InteractionAgentRuntime().execute("yes send it"))
    assert sent == ["Mailer"]


def test_the_user_sees_a_draft_once(llm):
    body = "Hi Sam, the cat pickup is on Friday with John."
    draft = {"to": "sam@example.test", "subject": "Cat pickup", "body": body}
    llm.script(
        completion("", [tool_call("send_draft", draft)]),
        completion("", [tool_call("send_message_to_user", {"message": f"{body}\n\nWant me to send this?"})]),
    )
    run(InteractionAgentRuntime().execute("email Sam about the cat pickup"))
    assert sum(reply.count(body) for reply in _replies()) == 1


def test_the_user_is_told_when_the_main_agent_cannot_answer(llm):
    llm.script(OpenRouterError("OpenRouter request failed (429): rate limited"))
    run(InteractionAgentRuntime().execute("remind me to stretch"))
    assert _replies(), "user-visible conversation is empty after a failed turn"


def test_a_failed_worker_run_is_reported_not_raised(llm):
    llm.script(OpenRouterError("boom"))
    result = run(ExecutionAgentRuntime("Mailer").execute("do something"))
    assert result.success is False and "boom" in result.response


def test_a_draft_follow_up_keeps_the_question_and_drops_the_repeat(llm):
    body = "Hi Sam, the cat pickup is on Friday with John."
    llm.script(completion("", [tool_call("send_draft", {"to": "sam@example.test", "subject": "Cat pickup", "body": body})]),
               completion("", [tool_call("send_message_to_user",
                                         {"message": f"To: sam@example.test\nSubject: Cat pickup\n\n{body}\n\nWant me to send this?"})]))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] Email to Sam: draft created"))
    replies = _replies()
    assert "Want me to send this?" in replies and sum(body in r for r in replies) == 1
    assert not any(r.startswith("To:") and "Want me" in r for r in replies)  # the question is not glued to a copy


def test_a_follow_up_that_only_repeats_the_draft_is_sent_back_to_the_model(llm):
    body = "Hi Sam, the cat pickup is on Friday with John."
    llm.script(completion("", [tool_call("send_draft", {"to": "sam@example.test", "subject": "Cat pickup", "body": body})]),
               completion("", [tool_call("send_message_to_user", {"message": body})]),
               completion("", [tool_call("send_message_to_user", {"message": "Send it?"})]))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] Email to Sam: draft created"))
    replies = _replies()
    assert sum(body in r for r in replies) == 1 and replies[-1] == "Send it?"


def test_quoting_a_draft_from_an_earlier_turn_is_left_alone(llm):
    body = "Hi Sam, the cat pickup is on Friday with John."
    llm.script(completion("", [tool_call("send_message_to_user", {"message": f"Earlier I drafted: {body}"})]))
    run(InteractionAgentRuntime().execute("what did you write to Sam?"))
    assert _replies() == [f"Earlier I drafted: {body}"]


def test_a_failed_task_result_turn_also_leaves_a_note(llm):
    llm.script(OpenRouterError("OpenRouter request failed (402): out of credits"))
    run(InteractionAgentRuntime().handle_agent_message("[SUCCESS] Garbage Reminder: it is time"))
    assert any("finished" in reply for reply in _replies())


def test_a_failed_email_alert_turn_stays_quiet(llm):
    llm.script(OpenRouterError("OpenRouter request failed (429)"))
    run(InteractionAgentRuntime().handle_agent_message("Important email watcher notification:\nInvoice due Friday"))
    assert _replies() == []


def test_the_failure_note_does_not_leak_the_error_details(llm):
    llm.script(OpenRouterError("OpenRouter request failed (402): key sk-or-v1-SECRET out of credits"))
    run(InteractionAgentRuntime().execute("hello"))
    assert all("SECRET" not in reply and "402" not in reply for reply in _replies())


def test_tool_arguments_that_are_not_a_json_object_are_reported_too(llm, gmail):
    llm.script(completion("", [tool_call("gmail_reply_to_thread", "[1, 2, 3]")]), completion("done"))
    run(ExecutionAgentRuntime("Mailer").execute("reply"))
    assert "json object" in _tool_message_text(llm.calls[1])
    assert gmail.calls == []  # the tool was never run with guessed arguments
