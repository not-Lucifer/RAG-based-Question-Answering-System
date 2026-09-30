"""Chat operations: ask (with history), streaming, sessions, feedback."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.chains import rag_chain
from app.core.config import get_settings
from app.core.exceptions import AppError, NotFoundError
from app.core.logging import get_logger
from app.db import crud
from app.db.database import SessionLocal
from app.db.models import ChatSession, Message
from app.schemas.chat import AskRequest, AskResponse, MessageOut, SessionOut, SourceOut

log = get_logger(__name__)
TITLE_CHARS = 60


def make_title(question: str) -> str:
    """Session title from the first question."""
    q = " ".join(question.split())
    return q if len(q) <= TITLE_CHARS else q[: TITLE_CHARS - 1].rstrip() + "…"


def _source_dicts(result: rag_chain.RAGResult) -> list[dict[str, Any]]:
    return [SourceOut(**s.to_dict()).model_dump() for s in result.sources]


def _message_out(msg: Message) -> MessageOut:
    sources = [SourceOut(**s) for s in json.loads(msg.sources_json)] if msg.sources_json else []
    return MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        sources=sources,
        latency_ms=msg.latency_ms,
        created_at=msg.created_at,
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class ChatService:
    """Business logic for chat. Owns DB transactions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def _session_and_history(self, session_id: str | None) -> tuple[ChatSession | None, list[Message]]:
        if not session_id:
            return None, []
        session = crud.get_session(self.db, session_id)
        if session is None:
            raise NotFoundError(f"Chat session '{session_id}' not found.")
        turns = get_settings().memory.history_turns
        return session, crud.get_messages(self.db, session_id, limit=2 * turns)

    def _persist(
        self, session: ChatSession | None, question: str, result: rag_chain.RAGResult, latency_ms: int
    ) -> tuple[ChatSession, Message]:
        if session is None:
            session = crud.create_session(self.db, make_title(question))
        crud.add_message(self.db, session.id, "user", question)
        msg = crud.add_message(
            self.db,
            session.id,
            "assistant",
            result.answer,
            sources=_source_dicts(result),
            latency_ms=latency_ms,
        )
        session.updated_at = datetime.now(UTC)
        self.db.commit()
        return session, msg

    def ask(self, req: AskRequest) -> AskResponse:
        """Answer a question; creates the session on the first question."""
        session, history = self._session_and_history(req.session_id)
        started = time.perf_counter()
        result = rag_chain.answer(req.question, history, req.filters(), req.top_k, req.mode)
        latency_ms = int((time.perf_counter() - started) * 1000)
        session, msg = self._persist(session, req.question, result, latency_ms)
        return AskResponse(
            answer=result.answer,
            sources=[SourceOut(**s) for s in _source_dicts(result)],
            session_id=session.id,
            latency_ms=latency_ms,
            message_id=msg.id,
            used_context=result.used_context,
            standalone_question=result.standalone_question,
        )

    def stream(self, req: AskRequest) -> Iterator[str]:
        """Server-Sent Events: ``token`` events, then one ``sources`` event (or ``error``)."""
        session, history = self._session_and_history(req.session_id)
        session_id = session.id if session else None
        history_data = [{"role": m.role, "content": m.content} for m in history]
        self.db.close()  # the generator outlives the request-scoped session

        def events() -> Iterator[str]:
            started = time.perf_counter()
            try:
                result = None
                for kind, payload in rag_chain.stream_answer(
                    req.question, history_data, req.filters(), req.top_k, req.mode
                ):
                    if kind == "token":
                        yield _sse("token", {"t": payload})
                    else:
                        result = payload
                assert result is not None
                latency_ms = int((time.perf_counter() - started) * 1000)
                with SessionLocal() as db:
                    svc = ChatService(db)
                    sess = crud.get_session(db, session_id) if session_id else None
                    sess, msg = svc._persist(sess, req.question, result, latency_ms)
                    final = AskResponse(
                        answer=result.answer,
                        sources=[SourceOut(**s) for s in _source_dicts(result)],
                        session_id=sess.id,
                        latency_ms=latency_ms,
                        message_id=msg.id,
                        used_context=result.used_context,
                        standalone_question=result.standalone_question,
                    )
                yield _sse("sources", final.model_dump(mode="json"))
            except AppError as exc:
                yield _sse("error", {"error": exc.code, "detail": exc.detail})
            except Exception:
                log.exception("stream failed")
                yield _sse("error", {"error": "INTERNAL_ERROR", "detail": "Unexpected server error."})

        return events()

    def list_sessions(self) -> list[SessionOut]:
        """All sessions, most recent first."""
        return [
            SessionOut(id=s.id, title=s.title, created_at=s.created_at, updated_at=s.updated_at)
            for s in crud.list_sessions(self.db)
        ]

    def get_messages(self, session_id: str) -> list[MessageOut]:
        """All messages of a session."""
        if crud.get_session(self.db, session_id) is None:
            raise NotFoundError(f"Chat session '{session_id}' not found.")
        return [_message_out(m) for m in crud.get_messages(self.db, session_id)]

    def delete_session(self, session_id: str) -> None:
        """Delete a session with its messages and feedback."""
        session = crud.get_session(self.db, session_id)
        if session is None:
            raise NotFoundError(f"Chat session '{session_id}' not found.")
        self.db.delete(session)
        self.db.commit()

    def add_feedback(self, message_id: int, rating: int, comment: str | None) -> None:
        """Store a rating for an assistant message."""
        msg = crud.get_message(self.db, message_id)
        if msg is None or msg.role != "assistant":
            raise NotFoundError(f"Assistant message {message_id} not found.")
        crud.add_feedback(self.db, message_id, rating, comment)
        self.db.commit()
