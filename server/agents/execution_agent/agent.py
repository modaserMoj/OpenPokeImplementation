"""Execution Agent implementation."""

from pathlib import Path
from typing import List, Optional, Dict, Any

from ...services.execution import get_execution_agent_logs
from ...services.execution import memory
from ...logging_config import logger


# Load system prompt template from file
_prompt_path = Path(__file__).parent / "system_prompt.md"
if _prompt_path.exists():
    SYSTEM_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8").strip()
else:
    # Placeholder template - you'll replace this with actual instructions
    SYSTEM_PROMPT_TEMPLATE = """You are an execution agent responsible for completing specific tasks using available tools.

Agent Name: {agent_name}
Purpose: {agent_purpose}

Instructions:
[TO BE FILLED IN BY USER]

You have access to Gmail tools to help complete your tasks. When given instructions:
1. Analyze what needs to be done
2. Use the appropriate tools to complete the task
3. Provide clear status updates on your actions

Be thorough, accurate, and efficient in your execution."""


class ExecutionAgent:
    """Manages state and history for an execution agent."""

    # Initialize execution agent with name, conversation limits, and log store access
    def __init__(
        self,
        name: str,
        conversation_limit: Optional[int] = None
    ):
        """
        Initialize an execution agent.

        Args:
            name: Human-readable agent name (e.g., 'conversation with keith')
            conversation_limit: Optional limit on past conversations to include (None = all)
        """
        self.name = name
        self.conversation_limit = conversation_limit
        self._log_store = get_execution_agent_logs()

    # Generate system prompt template with agent name and purpose derived from name
    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        agent_purpose = f"Handle tasks related to: {self.name}"

        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=agent_purpose
        )

    # Combine base system prompt with the agent's bounded memory (summary + identifiers + recent runs)
    def build_system_blocks(self) -> List[str]:
        """[stable base prompt, this agent's memory]: two blocks so each can be cached on its own."""
        history = memory.render_history(self.name)
        blocks = [self.build_system_prompt()]
        if history:
            blocks.append(f"# Execution History\n\n{history}")
        return blocks

    def build_system_prompt_with_history(self) -> str:
        """System prompt plus the agent's memory, never more than memory.HISTORY_BUDGET_CHARS of history.

        The full log stays on disk; see server/services/execution/memory.py."""
        return "\n\n".join(self.build_system_blocks())

    # Format current instruction as user message for LLM consumption
    def build_messages_for_llm(self, current_instruction: str) -> List[Dict[str, str]]:
        """
        Build message array for LLM call.

        Args:
            current_instruction: Current instruction from interaction agent

        Returns:
            List of messages in OpenRouter format
        """
        return [
            {"role": "user", "content": current_instruction}
        ]

    # Log the request this run is answering (once, by the run itself, so reminder runs are recorded too)
    def record_request(self, instructions: str) -> None:
        self._log_store.record_request(self.name, instructions)

    # Log the agent's final response to the execution log store
    def record_response(self, response: str) -> None:
        """Record agent's response to the log."""
        self._log_store.record_agent_response(self.name, response)

    # Log a tool call: full arguments, a digest of the result, the full result kept in a side file
    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> str:
        """Record tool execution details. Returns the stored result id ('' when the result was short)."""
        return memory.record_tool_call(self.name, tool_name, arguments, result)
