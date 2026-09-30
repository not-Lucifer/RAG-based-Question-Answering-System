"""FastAPI dependencies: DB session and services."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


def db_session() -> Iterator[Session]:
    """Request-scoped DB session."""
    yield from get_db()


def document_service(db: Session = Depends(db_session)) -> DocumentService:
    """DocumentService bound to the request session."""
    return DocumentService(db)


def chat_service(db: Session = Depends(db_session)) -> ChatService:
    """ChatService bound to the request session."""
    return ChatService(db)
