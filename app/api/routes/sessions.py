"""Chat session and feedback endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import chat_service
from app.schemas.chat import FeedbackRequest, MessageOut, OkResponse, SessionOut
from app.schemas.document import DeleteResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["sessions"])


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(svc: ChatService = Depends(chat_service)) -> list[SessionOut]:
    """All chat sessions, most recent first."""
    return svc.list_sessions()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_messages(session_id: str, svc: ChatService = Depends(chat_service)) -> list[MessageOut]:
    """Messages of a session, oldest first."""
    return svc.get_messages(session_id)


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
def delete_session(session_id: str, svc: ChatService = Depends(chat_service)) -> DeleteResponse:
    """Delete a session and its messages."""
    svc.delete_session(session_id)
    return DeleteResponse()


@router.post("/feedback", response_model=OkResponse)
def feedback(req: FeedbackRequest, svc: ChatService = Depends(chat_service)) -> OkResponse:
    """Rate an assistant answer (+1 / -1)."""
    svc.add_feedback(req.message_id, req.rating, req.comment)
    return OkResponse()
