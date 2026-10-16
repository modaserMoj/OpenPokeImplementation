"""Run execution agents and hand their results to the outbox.

Before: results were held until EVERY agent started in the same batch had finished (one slow agent delayed all of
them, up to the 90-second timeout), and two requests to the same agent could run at the same time without seeing
each other. Now:
  * each result goes to the outbox as soon as it is ready (the outbox merges results that finish together), and
  * runs of the same agent are queued, so the second one sees the first one's result. Different agents still run
    in parallel.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .runtime import ExecutionAgentRuntime, ExecutionResult
from ...logging_config import logger
from ...services.execution.roster import normalize_name
from ...services.outbox import REQUESTED, get_outbox


@dataclass
class PendingExecution:
    """Track a pending execution request."""

    request_id: str
    agent_name: str
    instructions: str
    kind: str
    created_at: datetime = field(default_factory=datetime.now)


_agent_locks: Dict[Tuple[int, str], asyncio.Lock] = {}


def _agent_lock(agent_name: str) -> asyncio.Lock:
    """One queue per agent (shared by every manager: user requests and reminders alike)."""
    key = (id(asyncio.get_running_loop()), normalize_name(agent_name))
    if key not in _agent_locks:
        _agent_locks[key] = asyncio.Lock()
    return _agent_locks[key]


def format_result(result: ExecutionResult) -> str:
    """How one finished run is reported to the main agent."""
    status = "SUCCESS" if result.success else "FAILED"
    response_text = (result.response or "(no response provided)").strip()
    return f"[{status}] {result.agent_name}: {response_text}"


class ExecutionBatchManager:
    """Run execution agents and report each result through the outbox."""

    def __init__(self, timeout_seconds: int = 90, kind: str = REQUESTED) -> None:
        self.timeout_seconds = timeout_seconds
        self.kind = kind  # outbox kind: "requested" for user requests, "reminder" for triggers
        self._pending: Dict[str, PendingExecution] = {}

    async def execute_agent(
        self,
        agent_name: str,
        instructions: str,
        request_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Run the agent (after any earlier run of the same agent), then hand the result to the outbox."""
        request_id = request_id or str(uuid.uuid4())
        self._pending[request_id] = PendingExecution(request_id, agent_name, instructions, self.kind)
        try:
            async with _agent_lock(agent_name):
                result = await self._run(agent_name, instructions)
        finally:
            self._pending.pop(request_id, None)
        get_outbox().add(self.kind, format_result(result))
        return result

    async def _run(self, agent_name: str, instructions: str) -> ExecutionResult:
        try:
            logger.info(f"[{agent_name}] Execution started")
            runtime = ExecutionAgentRuntime(agent_name=agent_name)
            result = await asyncio.wait_for(runtime.execute(instructions), timeout=self.timeout_seconds)
            logger.info(f"[{agent_name}] Execution finished: {'SUCCESS' if result.success else 'FAILED'}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[{agent_name}] Execution timed out after {self.timeout_seconds}s")
            return ExecutionResult(agent_name=agent_name, success=False,
                                   response=f"Execution timed out after {self.timeout_seconds} seconds", error="Timeout")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"[{agent_name}] Execution failed unexpectedly")
            return ExecutionResult(agent_name=agent_name, success=False, response=f"Execution failed: {exc}", error=str(exc))

    def get_pending_executions(self) -> List[Dict[str, object]]:
        """Expose pending executions for observability."""
        return [
            {
                "request_id": p.request_id,
                "agent_name": p.agent_name,
                "kind": p.kind,
                "created_at": p.created_at.isoformat(),
                "elapsed_seconds": (datetime.now() - p.created_at).total_seconds(),
            }
            for p in self._pending.values()
        ]

    async def shutdown(self) -> None:
        """Clear pending bookkeeping (results still waiting in the outbox are flushed by the app on shutdown)."""
        self._pending.clear()
