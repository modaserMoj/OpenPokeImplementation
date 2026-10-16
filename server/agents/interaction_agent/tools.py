"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from ...logging_config import logger
from ...services.approvals import current_turn, get_approval_store
from ...services.conversation import get_conversation_log
from ...services.execution import get_agent_roster, get_execution_agent_logs
from ...services.execution import lifecycle
from ..execution_agent.batch_manager import ExecutionBatchManager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Deliver instructions to a specific execution agent. Creates a new agent if the name doesn't exist in the roster, or reuses an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable agent name describing its purpose (e.g., 'Vercel Job Offer', 'Email to Sharanjeet'). This name will be used to identify and potentially reuse the agent."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["agent_name", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Show the user the exact email (all recipients, subject, body) so they can review it. Only text shown with this tool can be approved for sending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Every recipient of the draft (to, cc and bcc), exactly as it will be sent.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_send",
            "description": "Record the user's approval to send a draft you already showed them with send_draft. Call it only when the user's latest message clearly confirms (for example 'yes, send it'). The system refuses it otherwise. After it succeeds, tell the agent to send exactly that content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {
                        "type": "string",
                        "description": "Which draft, only needed when more than one is waiting for approval.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()


# Create or reuse execution agent and dispatch instructions asynchronously
def send_message_to_agent(agent_name: str, instructions: str) -> ToolResult:
    """Send instructions to an execution agent.

    A known name (or an alias, or the same name spelled differently) goes straight to that agent. A name nobody
    has used yet is first checked against the existing agents in the background: if one already covers this
    thread it is reused (the new name becomes its alias); otherwise a new agent is created. See
    server/services/execution/lifecycle.py."""
    roster = get_agent_roster()
    roster.load()
    requested_name = agent_name
    existing = roster.resolve(agent_name)
    is_new = existing is None
    if existing is not None:
        agent_name = existing.name
        roster.touch(agent_name)
        logger.info(f"Reused agent: {agent_name}")

    async def _execute_async() -> None:
        target = agent_name
        try:
            if is_new:
                target = await lifecycle.resolve_new_agent(requested_name, instructions)
                roster.touch(target)
                logger.info(f"{'Created' if target == requested_name else 'Reused'} agent: {target}")
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(target, instructions)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{target}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{target}' failed: {str(exc)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    loop.create_task(_execute_async())

    payload = {"status": "submitted", "agent_name": agent_name, "new_agent_created": is_new}
    if requested_name != agent_name:
        payload["resolved_from"] = requested_name
    if is_new:
        payload["note"] = ("No agent has this name yet. If an existing agent already covers this thread it is reused; "
                           "the result will say which agent handled it.")
    return ToolResult(success=True, payload=payload)


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log.

    If the message repeats a draft the user was just shown with send_draft in this same turn, the repeated part is
    removed, so the user sees the draft once."""
    turn = current_turn()
    cleaned = _without_shown_drafts(message, turn.drafts) if turn else message
    if cleaned != message and not cleaned.strip():
        return ToolResult(
            success=False,
            payload={"error": "That only repeats the draft the user was just shown. Send a short question asking whether to send it."},
        )
    log = get_conversation_log()
    log.record_reply(cleaned)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=cleaned,
        recorded_reply=True,
    )


def _plain(text: str) -> str:
    return " ".join((text or "").lower().split())


def _without_shown_drafts(message: str, drafts) -> str:
    """Drop the paragraphs of `message` that are just part of a draft already shown in this turn."""
    if not drafts:
        return message
    shown = " ".join(_plain(f"to: {to} subject: {subject} {body}") for to, subject, body in drafts)
    kept = [p for p in message.split("\n\n") if not (_plain(p) and _plain(p) in shown)]
    return "\n\n".join(kept).strip() if len(kept) != len(message.split("\n\n")) else message


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    log = get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")
    store = get_approval_store()
    shown = store.mark_shown(to, body)
    if not shown:  # nothing an agent created matches: the shown text itself becomes approvable (sent word for word)
        store.record_shown(to, subject, body)
    turn = current_turn()
    if turn is not None:
        turn.drafts.append((to, subject, body))

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
            "can_be_approved": [p.id for p in shown],
        },
        recorded_reply=True,
    )


# Record that the user approved sending a draft they were shown (enforced by code, see services/approvals.py)
def approve_send(proposal_id: Optional[str] = None) -> ToolResult:
    proposal, reason = get_approval_store().approve(proposal_id)
    if proposal is None:
        return ToolResult(success=False, payload={"error": reason})
    return ToolResult(
        success=True,
        payload={
            "status": "approved",
            **proposal.label(),
            "next": "Tell the agent to send it now, with exactly the same recipients and text.",
        },
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "approve_send":
            return approve_send(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
