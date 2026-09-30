"""Vector retrievers (similarity / MMR) with metadata filters and relevance scores.

Every returned Document carries ``metadata["score"]``: the cosine similarity
between the query and the chunk embedding (0..1 for normalised embeddings). The
RAG chain compares the best score with ``retrieval.score_threshold``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import get_settings
from app.core.exceptions import AppError, EmbeddingError
from app.core.logging import get_logger
from app.embeddings.factory import get_embeddings
from app.vectorstore.store import Candidate, query_candidates

log = get_logger(__name__)
Mode = Literal["similarity", "mmr", "hybrid"]


def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map API filters to a Chroma ``where`` clause.

    ``{"subject": "DBMS"}`` or ``{"subject": ["DBMS", "OS"]}`` filter by subject;
    ``{"doc_ids": [...]}`` (or ``doc_id``) restricts to documents. Both are ANDed.
    """
    if not filters:
        return None
    clauses: list[dict[str, Any]] = []
    subject = filters.get("subject") or filters.get("subjects")
    if subject:
        subjects = [subject] if isinstance(subject, str) else list(subject)
        clauses.append({"subject": subjects[0]} if len(subjects) == 1 else {"subject": {"$in": subjects}})
    doc_ids = filters.get("doc_ids") or filters.get("doc_id")
    if doc_ids:
        ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)
        clauses.append({"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def embed_query(text: str) -> list[float]:
    """Embed a query, mapping provider failures to EmbeddingError."""
    try:
        return list(get_embeddings().embed_query(text))
    except AppError:
        raise
    except Exception as exc:
        log.error("query embedding failed", extra={"fields": {"error": type(exc).__name__}})
        raise EmbeddingError("Could not embed the question.") from exc


def cosine_scores(query: list[float], vectors: list[list[float]]) -> np.ndarray:
    """Cosine similarity of ``query`` against each row of ``vectors``."""
    if not vectors:
        return np.zeros(0)
    q = np.asarray(query, dtype=float)
    m = np.asarray(vectors, dtype=float)
    denom = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
    denom[denom == 0] = 1.0
    return (m @ q) / denom


def mmr_select(query: list[float], vectors: list[list[float]], k: int, lambda_mult: float) -> list[int]:
    """Maximal Marginal Relevance: indices of ``k`` relevant yet diverse vectors."""
    if not vectors:
        return []
    relevance = cosine_scores(query, vectors)
    m = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = m / norms
    selected = [int(np.argmax(relevance))]
    while len(selected) < min(k, len(vectors)):
        redundancy = (unit @ unit[selected].T).max(axis=1)
        mmr = lambda_mult * relevance - (1 - lambda_mult) * redundancy
        mmr[selected] = -np.inf
        selected.append(int(np.argmax(mmr)))
    return selected


def _annotate(cand: Candidate, score: float) -> Document:
    meta = dict(cand.document.metadata)
    meta.update({"score": round(float(score), 4), "chunk_id": cand.id})
    return Document(page_content=cand.document.page_content, metadata=meta)


class VectorRetriever(BaseRetriever):
    """Chroma retriever supporting similarity and MMR, returning scored Documents."""

    mode: Literal["similarity", "mmr"] = "mmr"
    top_k: int = 5
    fetch_k: int = 20
    mmr_lambda: float = 0.6
    where: dict[str, Any] | None = None

    def search_by_embedding(self, query_emb: list[float], k: int | None = None) -> list[Document]:
        """Retrieve using a precomputed query embedding."""
        k = k or self.top_k
        n = max(self.fetch_k, k) if self.mode == "mmr" else k
        cands = query_candidates(query_emb, n, self.where)
        if not cands:
            return []
        scores = cosine_scores(query_emb, [c.embedding for c in cands])
        if self.mode == "mmr":
            order = mmr_select(query_emb, [c.embedding for c in cands], k, self.mmr_lambda)
        else:
            order = [int(i) for i in np.argsort(-scores)[:k]]
        return [_annotate(cands[i], scores[i]) for i in order]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.search_by_embedding(embed_query(query))


def build_retriever(
    mode: Mode | None = None, top_k: int | None = None, filters: dict[str, Any] | None = None
) -> BaseRetriever:
    """Build a retriever for ``mode`` (defaults from settings).

    ``similarity`` = plain cosine top-k; ``mmr`` = diverse top-k from ``fetch_k``
    candidates; ``hybrid`` = BM25 + vector with weighted rank fusion.
    """
    cfg = get_settings().retrieval
    mode = mode or cfg.mode
    top_k = top_k or cfg.top_k
    where = build_where(filters)
    if mode == "hybrid":
        from app.retrieval.hybrid import build_hybrid

        vector = VectorRetriever(mode="similarity", top_k=top_k, fetch_k=cfg.fetch_k, where=where)
        return build_hybrid(vector, filters)
    if mode not in ("similarity", "mmr"):
        raise ValueError(f"Unknown retrieval mode: {mode}")
    return VectorRetriever(
        mode=mode, top_k=top_k, fetch_k=cfg.fetch_k, mmr_lambda=cfg.mmr_lambda, where=where
    )
