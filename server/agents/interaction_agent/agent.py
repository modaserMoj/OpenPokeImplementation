"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...services.execution import get_agent_roster
from ...utils.timezones import current_time_note
from ...services.execution.retrieval import aselect_agents, select_agents

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    active_agents: Optional[str] = None,
    notes: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn.

    `active_agents` is the pre-rendered agent section from `render_active_agents` (which can use the semantic
    ranker). When omitted, a word-matching-only view is built synchronously."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{active_agents if active_agents is not None else _render_active_agents(latest_text)}\n</active_agents>")
    sections.append(f"<current_time>{current_time_note()}</current_time>")
    if notes:
        sections.append(notes)
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _format_agents(records, shown) -> str:
    rendered: List[str] = []
    for record in shown:
        name = escape(record.name or "agent", quote=True)
        about = escape(" ".join(record.description.split())[:100], quote=True)
        attributes = f'name="{name}"' + (f' about="{about}"' if about else "")
        # Who the agent deals with, and the other names it answers to (duplicates merged into it): without these
        # the model cannot tell that "any word from Lewis?" belongs to "Email to Zeke" (Zeke Lewis).
        if record.people:
            attributes += f' people="{escape(", ".join(record.people[:4]), quote=True)}"'
        if record.aliases:
            attributes += f' also_called="{escape(", ".join(record.aliases[:3]), quote=True)}"'
        attributes += f' last_active="{record.last_active_at[:10]}"'
        rendered.append(f"<agent {attributes} />")
    hidden = len(records) - len(shown)
    if hidden > 0:
        rendered.append(f'<other_agents count="{hidden}" note="not listed because they match this message less well" />')
    return "\n".join(rendered)


# Format the execution agents relevant to this message (word matching only; used when no async view is given)
def _render_active_agents(query: str = "") -> str:
    roster = get_agent_roster()
    roster.load()
    records = roster.list_records()
    if not records:
        return "None"
    return _format_agents(records, select_agents(records, query))


# Format the execution agents relevant to this message, using every configured ranker (word + semantic)
async def render_active_agents(query: str = "") -> str:
    roster = get_agent_roster()
    roster.load()
    records = roster.list_records()
    if not records:
        return "None"
    return _format_agents(records, await aselect_agents(records, query))


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
