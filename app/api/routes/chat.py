"""Chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import chat_service
from app.schemas.chat import AskRequest, AskResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, svc: ChatService = Depends(chat_service)) -> AskResponse:
    """Answer a question with citations."""
    return svc.ask(req)


@router.post("/stream")
def ask_stream(req: AskRequest, svc: ChatService = Depends(chat_service)) -> StreamingResponse:
    """Same as /chat/ask, streamed as Server-Sent Events (``token`` ... ``sources`` | ``error``)."""
    return StreamingResponse(
        svc.stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
