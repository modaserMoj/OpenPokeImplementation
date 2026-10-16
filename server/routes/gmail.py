from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..config import Settings, get_settings
from ..models import GmailConnectPayload, GmailDisconnectPayload, GmailStatusPayload
from ..services import disconnect_account, fetch_status, initiate_connect

router = APIRouter(prefix="/gmail", tags=["gmail"])


@router.post("/connect")
# Initiate Gmail OAuth connection flow through Composio
async def gmail_connect(payload: GmailConnectPayload, settings: Settings = Depends(get_settings)) -> JSONResponse:
    return await asyncio.to_thread(initiate_connect, payload, settings)  # Composio calls are synchronous


@router.post("/status")
# Check the current Gmail connection status and user information
async def gmail_status(payload: GmailStatusPayload) -> JSONResponse:
    return await asyncio.to_thread(fetch_status, payload)


@router.post("/disconnect")
# Disconnect Gmail account and clear cached profile data
async def gmail_disconnect(payload: GmailDisconnectPayload) -> JSONResponse:
    return await asyncio.to_thread(disconnect_account, payload)
