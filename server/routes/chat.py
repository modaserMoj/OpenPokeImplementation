from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import ChatHistoryClearResponse, ChatHistoryResponse, ChatRequest
from ..services import get_conversation_log, get_trigger_service, handle_chat_request

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/send", response_class=JSONResponse, summary="Submit a chat message and receive a completion")
# Handle incoming chat messages and route them to the interaction agent
async def chat_send(
    payload: ChatRequest,
) -> JSONResponse:
    return await handle_chat_request(payload)


@router.get("/history", response_model=ChatHistoryResponse)
# Retrieve the conversation history from the log
def chat_history() -> ChatHistoryResponse:
    log = get_conversation_log()
    return ChatHistoryResponse(messages=log.to_chat_messages())


@router.delete("/history", response_model=ChatHistoryClearResponse)
def clear_history(everything: bool = False) -> ChatHistoryClearResponse:
    """Clear the chat transcript (and the drafts shown in it, which can no longer be approved). Agents, their memory,
    stored results and reminders are KEPT: before, this also
    deleted every agent and every reminder (and missed newer stores, leaving orphaned files).

    `?everything=true` is an explicit full reset that clears every store consistently."""
    log = get_conversation_log()
    log.clear()
    # Drafts shown in the cleared chat are no longer visible, so they must not be approvable: a later "yes" would
    # otherwise send an email the user can no longer see (found in live test 7).
    from ..services.approvals import get_approval_store

    get_approval_store().clear()
    if everything:
        _reset_everything()
    return ChatHistoryClearResponse()


def _reset_everything() -> None:
    import shutil

    from ..services import get_agent_roster, get_execution_agent_logs
    from ..services.approvals import get_approval_store

    logs = get_execution_agent_logs()
    logs.clear_all()
    base = logs._base_dir
    for pattern in ("*.memory.json", "embeddings_*.json"):
        for path in base.glob(pattern):
            path.unlink(missing_ok=True)
    shutil.rmtree(base / "results", ignore_errors=True)
    get_agent_roster().clear()
    get_trigger_service().clear_all()
    get_approval_store().clear()


__all__ = ["router"]
