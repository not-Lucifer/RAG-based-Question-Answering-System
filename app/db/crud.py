"""Data-access helpers. Callers own the transaction (commit/rollback)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ChatSession, DocumentRecord, DocumentStatus, Feedback, Message

# ---------------------------------------------------------------- documents


def get_document(db: Session, doc_id: str) -> DocumentRecord | None:
    """Fetch a document by id."""
    return db.get(DocumentRecord, doc_id)


def get_document_by_hash(db: Session, file_hash: str) -> DocumentRecord | None:
    """Fetch a document by SHA-256 content hash."""
    return db.scalar(select(DocumentRecord).where(DocumentRecord.file_hash == file_hash))


def list_documents(db: Session) -> Sequence[DocumentRecord]:
    """All documents, newest first."""
    return db.scalars(select(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc())).all()


def count_ready_documents(db: Session) -> int:
    """Number of documents with status READY."""
    stmt = (
        select(func.count())
        .select_from(DocumentRecord)
        .where(DocumentRecord.status == DocumentStatus.READY.value)
    )
    return int(db.scalar(stmt) or 0)


def create_document(db: Session, doc_id: str, filename: str, subject: str, file_hash: str) -> DocumentRecord:
    """Insert a PROCESSING document row."""
    record = DocumentRecord(
        id=doc_id,
        filename=filename,
        subject=subject,
        file_hash=file_hash,
        status=DocumentStatus.PROCESSING.value,
    )
    db.add(record)
    db.flush()
    return record


def mark_document_ready(db: Session, record: DocumentRecord, num_pages: int, num_chunks: int) -> None:
    """Set status READY with counts."""
    record.status = DocumentStatus.READY.value
    record.num_pages, record.num_chunks, record.error_message = num_pages, num_chunks, None


def mark_document_failed(db: Session, record: DocumentRecord, error_message: str) -> None:
    """Set status FAILED with a user-safe error message."""
    record.status = DocumentStatus.FAILED.value
    record.num_chunks = 0
    record.error_message = error_message[:1000]


# ---------------------------------------------------------------- chat


def get_session(db: Session, session_id: str) -> ChatSession | None:
    """Fetch a chat session by id."""
    return db.get(ChatSession, session_id)


def list_sessions(db: Session) -> Sequence[ChatSession]:
    """All sessions, most recently active first."""
    return db.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc())).all()


def create_session(db: Session, title: str) -> ChatSession:
    """Insert a session titled from the first question."""
    session = ChatSession(title=title)
    db.add(session)
    db.flush()
    return session


def get_messages(db: Session, session_id: str, limit: int | None = None) -> list[Message]:
    """Messages of a session, oldest first (``limit`` keeps only the latest N)."""
    stmt = select(Message).where(Message.session_id == session_id).order_by(Message.id.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(reversed(db.scalars(stmt).all()))


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    latency_ms: int | None = None,
) -> Message:
    """Insert a message; ``sources`` are stored as JSON."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources_json=json.dumps(sources) if sources is not None else None,
        latency_ms=latency_ms,
    )
    db.add(msg)
    db.flush()
    return msg


def get_message(db: Session, message_id: int) -> Message | None:
    """Fetch a message by id."""
    return db.get(Message, message_id)


def add_feedback(db: Session, message_id: int, rating: int, comment: str | None) -> Feedback:
    """Insert feedback for a message."""
    fb = Feedback(message_id=message_id, rating=rating, comment=comment)
    db.add(fb)
    db.flush()
    return fb
