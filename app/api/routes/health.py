"""Health endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import db_session
from app.core.config import get_settings
from app.db import crud
from app.vectorstore.store import index_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(db_session)) -> dict[str, Any]:
    """Liveness plus a cheap configuration summary (never loads models)."""
    s = get_settings()
    try:
        index = index_status()
    except Exception:
        index = {"ok": False, "stored_model": None, "current_model": s.active_embedding_model}
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,
        "llm_model": s.active_llm_model,
        "embedding_model": index["current_model"],
        "index_ok": index["ok"],
        "retrieval_mode": s.retrieval.mode,
        "top_k": s.retrieval.top_k,
        "docs_indexed": crud.count_ready_documents(db),
    }
