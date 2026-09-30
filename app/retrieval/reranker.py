"""Optional cross-encoder reranking (off by default; lazy-loaded)."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_model: Any | None = None
_model_name: str | None = None
_lock = threading.Lock()


def _load(model_name: str) -> Any:
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != model_name:
            from sentence_transformers import CrossEncoder  # heavy import, only when enabled

            log.info("loading reranker", extra={"fields": {"model": model_name}})
            _model, _model_name = CrossEncoder(model_name, device="cpu"), model_name
        return _model


def rerank(query: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    """Reorder ``docs`` by cross-encoder relevance and keep the best ``top_n``.

    Adds ``metadata["rerank_score"]``; the cosine ``score`` is kept for thresholding.
    On any model failure the original order is returned (reranking is best-effort).
    """
    cfg = get_settings().retrieval
    top_n = top_n or cfg.rerank_top_n
    if len(docs) <= 1:
        return docs[:top_n]
    try:
        model = _load(cfg.reranker_model)
        scores = model.predict([(query, d.page_content) for d in docs])
    except Exception as exc:
        log.warning("rerank failed; keeping retriever order", extra={"fields": {"error": type(exc).__name__}})
        return docs[:top_n]
    ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True)
    out = []
    for doc, score in ranked[:top_n]:
        meta = dict(doc.metadata)
        meta["rerank_score"] = round(float(score), 4)
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
