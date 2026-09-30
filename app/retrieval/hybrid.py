"""Hybrid retrieval: BM25 keyword search fused with vector search.

Uses weighted Reciprocal Rank Fusion (the same algorithm as LangChain's
``EnsembleRetriever``), implemented locally because ``EnsembleRetriever`` moved
between packages across LangChain releases. BM25 indexes are cached per filter
and invalidated automatically whenever the vector index changes (ingest/delete).
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import get_settings
from app.retrieval.retriever import VectorRetriever, build_where, cosine_scores, embed_query
from app.vectorstore.store import chunk_id, get_chunks, get_embeddings_by_ids, index_version

_RRF_K = 60
_TOKEN = re.compile(r"\w+", re.UNICODE)
_cache: dict[str, tuple[int, BM25Retriever | None]] = {}
_cache_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps terms like '3nf', 'tcp', 'o(n)' parts intact."""
    return _TOKEN.findall(text.lower())


def _doc_key(doc: Document) -> str:
    meta = doc.metadata
    return meta.get("chunk_id") or chunk_id(meta.get("doc_id", ""), meta.get("chunk_index", -1))


def get_bm25(where: dict[str, Any] | None, k: int) -> BM25Retriever | None:
    """Cached BM25 retriever over the (filtered) chunks; ``None`` if no chunks."""
    settings = get_settings()
    key = json.dumps([str(settings.chroma_dir), settings.chroma_collection, where], sort_keys=True)
    version = index_version()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is None or cached[0] != version:
            chunks = get_chunks(where)
            docs = []
            for cid, doc in chunks:
                doc.metadata["chunk_id"] = cid
                docs.append(doc)
            bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize) if docs else None
            _cache[key] = cached = (version, bm25)
    bm25 = cached[1]
    if bm25 is not None:
        bm25.k = k
    return bm25


def invalidate_bm25_cache() -> None:
    """Drop all cached BM25 indexes."""
    with _cache_lock:
        _cache.clear()


def weighted_rrf(result_lists: list[list[Document]], weights: list[float]) -> list[Document]:
    """Fuse ranked lists: score(d) = sum_i w_i / (RRF_K + rank_i(d))."""
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    for docs, weight in zip(result_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (_RRF_K + rank)
            first_seen.setdefault(key, doc)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [first_seen[k] for k in ordered]


class HybridRetriever(BaseRetriever):
    """BM25 + vector retriever with weighted rank fusion and cosine scores on every result."""

    vector: VectorRetriever
    where: dict[str, Any] | None = None
    weights: tuple[float, float] = (0.4, 0.6)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        k = self.vector.top_k
        pool = max(2 * k, 10)
        query_emb = embed_query(query)
        vector_docs = self.vector.search_by_embedding(query_emb, k=pool)
        bm25 = get_bm25(self.where, pool)
        keyword_docs = bm25.invoke(query) if bm25 is not None else []
        fused = weighted_rrf([keyword_docs, vector_docs], list(self.weights))[:k]

        # Keyword-only hits have no vector score yet: compute it from stored embeddings.
        missing = [_doc_key(d) for d in fused if "score" not in d.metadata]
        stored = get_embeddings_by_ids(missing)
        out = []
        for doc in fused:
            meta = dict(doc.metadata)
            key = _doc_key(doc)
            meta["chunk_id"] = key
            if "score" not in meta and key in stored:
                meta["score"] = round(float(cosine_scores(query_emb, [stored[key]])[0]), 4)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


def build_hybrid(vector_retriever: VectorRetriever, filters: dict[str, Any] | None) -> HybridRetriever:
    """Combine a vector retriever with a cached BM25 retriever over the same filtered chunks."""
    weights = get_settings().retrieval.hybrid_weights
    return HybridRetriever(vector=vector_retriever, where=build_where(filters), weights=tuple(weights))
