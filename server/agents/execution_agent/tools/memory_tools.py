"""Lets an execution agent read back the full output of an earlier tool call (its history keeps only a digest)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from server.services.execution.memory import ToolResultStore

MAX_RETURN_CHARS = 20_000

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_tool_result",
            "description": "Fetch the full output of an earlier tool call. Your history shows only a summary of long "
                           "results, ending with 'full result: r_xxxxxxxx'; pass that id.",
            "parameters": {
                "type": "object",
                "properties": {"result_id": {"type": "string", "description": "The id, for example r_1a2b3c4d."}},
                "required": ["result_id"],
                "additionalProperties": False,
            },
        },
    }
]


def get_schemas() -> List[Dict[str, Any]]:
    return _SCHEMAS


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:
    def get_tool_result(result_id: str) -> Dict[str, Any]:
        record = ToolResultStore().load(agent_name, result_id)
        if record is None:
            return {"error": f"No stored result {result_id!r} for this agent (it may have expired). Run the tool again."}
        text = record["result"]
        if len(text) > MAX_RETURN_CHARS:
            text = text[:MAX_RETURN_CHARS] + f"… [{len(record['result']):,} characters in total]"
        return {"tool": record["tool"], "arguments": record["arguments"], "result": text}

    return {"get_tool_result": get_tool_result}
