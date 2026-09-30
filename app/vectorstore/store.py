"""Persistent ChromaDB vector store.

Chunk ids are ``f"{doc_id}:{chunk_index}"`` so re-ingesting a document overwrites
it cleanly. The embedding model id is stored in the collection metadata; opening
the store with a different model raises :class:`IndexMismatchError`.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.exceptions import AppError, IndexMismatchError, VectorStoreError
from app.core.logging import get_logger
from app.embeddings.factory import embedding_model_id, get_embeddings

log = get_logger(__name__)

_lock = threading.RLock()
_clients: dict[str, Any] = {}
_stores: dict[tuple[str, str, str], Chroma] = {}
_index_version = 0  # bumped on every write; lets the BM25 cache invalidate itself


@dataclass(frozen=True)
class Candidate:
    """A chunk returned by a raw vector query, with its stored embedding."""

    id: str
    document: Document
    embedding: list[float]


def chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic chunk id."""
    return f"{doc_id}:{chunk_index}"


def index_version() -> int:
    """Monotonic counter of index writes in this process."""
    return _index_version


def _bump_version() -> None:
    global _index_version
    _index_version += 1


def _client() -> Any:
    path = str(get_settings().chroma_dir)
    with _lock:
        if path not in _clients:
            get_settings().chroma_dir.mkdir(parents=True, exist_ok=True)
            _clients[path] = chromadb.PersistentClient(
                path=path, settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
            )
        return _clients[path]


def _raw_collection() -> Any | None:
    """The Chroma collection, or ``None`` if it has not been created yet."""
    try:
        return _client().get_collection(get_settings().chroma_collection)
    except Exception:  # chromadb raises NotFoundError / ValueError depending on version
        return None


def index_status() -> dict[str, Any]:
    """Compare the stored embedding model with the configured one (no model loading)."""
    current = embedding_model_id()
    col = _raw_collection()
    stored = (col.metadata or {}).get("embedding_model") if col is not None else None
    return {"ok": stored in (None, current), "stored_model": stored, "current_model": current}


def check_index_compatibility() -> None:
    """Raise IndexMismatchError if the index was built with another embedding model."""
    status = index_status()
    if not status["ok"]:
        log.error(
            "embedding model mismatch",
            extra={"fields": {"stored": status["stored_model"], "current": status["current_model"]}},
        )
        raise IndexMismatchError(
            f"The index was built with '{status['stored_model']}' but the app is configured for "
            f"'{status['current_model']}'. Run `python scripts/reset_index.py --reindex` and try again."
        )


def get_vectorstore() -> Chroma:
    """Return the cached LangChain Chroma store for the current settings."""
    s = get_settings()
    key = (str(s.chroma_dir), s.chroma_collection, embedding_model_id())
    with _lock:
        if key in _stores:
            return _stores[key]
        check_index_compatibility()
        try:
            store = Chroma(
                client=_client(),
                collection_name=s.chroma_collection,
                embedding_function=get_embeddings(),
                collection_metadata={"embedding_model": key[2], "hnsw:space": "cosine"},
            )
        except AppError:
            raise
        except Exception as exc:
            log.exception("could not open vector store")
            raise VectorStoreError("Could not open the vector store.") from exc
        _stores[key] = store
        return store


def add_chunks(chunks: list[Document]) -> int:
    """Embed and upsert chunks in batches of ``embedding.batch_size``. Returns the count."""
    if not chunks:
        return 0
    store = get_vectorstore()
    batch = get_settings().embedding.batch_size
    try:
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            ids = [chunk_id(c.metadata["doc_id"], c.metadata["chunk_index"]) for c in part]
            store.add_documents(part, ids=ids)
    except AppError:
        raise
    except Exception as exc:
        log.exception("adding chunks failed")
        raise VectorStoreError("Storing document chunks failed.") from exc
    finally:
        _bump_version()
    return len(chunks)


def delete_doc(doc_id: str) -> int:
    """Delete all chunks of a document. Returns the number of chunks removed."""
    col = _raw_collection()
    if col is None:
        return 0
    try:
        ids = col.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            col.delete(ids=ids)
    except Exception as exc:
        log.exception("deleting chunks failed")
        raise VectorStoreError("Deleting document chunks failed.") from exc
    _bump_version()
    return len(ids)


def count() -> int:
    """Number of chunks in the collection."""
    col = _raw_collection()
    return 0 if col is None else int(col.count())


def get_chunks(where: dict[str, Any] | None = None) -> list[tuple[str, Document]]:
    """All stored chunks (optionally filtered) as ``(id, Document)`` pairs, for BM25."""
    col = _raw_collection()
    if col is None:
        return []
    try:
        res = col.get(where=where or None, include=["documents", "metadatas"])
    except Exception as exc:
        raise VectorStoreError("Reading chunks failed.") from exc
    return [
        (cid, Document(page_content=text or "", metadata=dict(meta or {})))
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=False)
    ]


def query_candidates(embedding: list[float], n: int, where: dict[str, Any] | None = None) -> list[Candidate]:
    """Nearest-neighbour query returning chunks together with their stored embeddings."""
    store = get_vectorstore()  # also validates model compatibility
    col = store._collection
    total = col.count()
    if total == 0:
        return []
    try:
        res = col.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            where=where or None,
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception as exc:
        log.exception("vector query failed")
        raise VectorStoreError("Vector search failed.") from exc
    return [
        Candidate(
            id=cid, document=Document(page_content=text or "", metadata=dict(meta or {})), embedding=list(emb)
        )
        for cid, text, meta, emb in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["embeddings"][0], strict=False
        )
    ]


def get_embeddings_by_ids(ids: list[str]) -> dict[str, list[float]]:
    """Stored embeddings for the given chunk ids."""
    col = _raw_collection()
    if col is None or not ids:
        return {}
    res = col.get(ids=ids, include=["embeddings"])
    return {cid: list(emb) for cid, emb in zip(res["ids"], res["embeddings"], strict=False)}


def list_doc_ids() -> set[str]:
    """Distinct doc_ids present in the index."""
    return {doc.metadata.get("doc_id", "") for _, doc in get_chunks()} - {""}


def reset_collection() -> None:
    """Drop the whole collection (used by scripts/reset_index.py)."""
    s = get_settings()
    with _lock:
        with contextlib.suppress(Exception):  # already absent
            _client().delete_collection(s.chroma_collection)
        _stores.clear()
        _bump_version()
    log.warning("vector collection reset", extra={"fields": {"collection": s.chroma_collection}})


def reset_caches() -> None:
    """Forget cached clients/stores (tests and evaluation switch directories)."""
    with _lock:
        _stores.clear()
        _clients.clear()
        _bump_version()
