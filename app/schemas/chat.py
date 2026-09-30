"""Chat, session and feedback API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """A question, optionally continuing a session and filtered by subject/documents."""

    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    subject: str | list[str] | None = None
    doc_ids: list[str] | None = None
    top_k: int | None = Field(None, ge=1, le=10)
    mode: Literal["similarity", "mmr", "hybrid"] | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    def filters(self) -> dict[str, Any] | None:
        """Retriever filters derived from ``subject`` / ``doc_ids``."""
        out: dict[str, Any] = {}
        if self.subject:
            out["subject"] = self.subject
        if self.doc_ids:
            out["doc_ids"] = self.doc_ids
        return out or None


class SourceOut(BaseModel):
    """A citation: ``[n] source · page``."""

    n: int
    source: str
    page: int
    snippet: str
    score: float | None = None
    doc_id: str | None = None
    subject: str | None = None


class AskResponse(BaseModel):
    """Answer with citations."""

    answer: str
    sources: list[SourceOut]
    session_id: str
    latency_ms: int
    message_id: int | None = None
    used_context: bool = True
    standalone_question: str | None = None


class SessionOut(BaseModel):
    """A chat session."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime | None = None


class MessageOut(BaseModel):
    """A stored message."""

    id: int
    role: str
    content: str
    sources: list[SourceOut] = []
    latency_ms: int | None = None
    created_at: datetime


class FeedbackRequest(BaseModel):
    """Thumbs up (1) or down (-1) on an assistant message."""

    message_id: int
    rating: Literal[1, -1]
    comment: str | None = Field(None, max_length=1000)


class OkResponse(BaseModel):
    """Generic acknowledgement."""

    ok: bool = True
