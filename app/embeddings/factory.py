"""Embedding model factory (local sentence-transformers or OpenAI)."""

from __future__ import annotations

import threading
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

log = get_logger(__name__)
_override: Embeddings | None = None
_build_lock = threading.Lock()  # startup warm-up and a first request must not load the model twice


def set_embeddings_override(embeddings: Embeddings | None) -> None:
    """Inject an Embeddings instance (tests / evaluation). ``None`` restores normal behaviour."""
    global _override
    _override = embeddings


@lru_cache
def _build_embeddings(provider: str, model: str, batch_size: int) -> Embeddings:
    settings = get_settings()
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            if settings.openai_api_key is None:
                raise EmbeddingError("OPENAI_API_KEY is not set; required for EMBEDDING_PROVIDER=openai.")
            return OpenAIEmbeddings(
                model=model, api_key=settings.openai_api_key, chunk_size=batch_size, max_retries=2
            )
        from langchain_huggingface import HuggingFaceEmbeddings

        log.info("loading local embedding model", extra={"fields": {"model": model}})
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        log.error("embedding model init failed", extra={"fields": {"error": type(exc).__name__}})
        raise EmbeddingError(f"Could not load embedding model '{model}'.") from exc


def get_embeddings() -> Embeddings:
    """Return the cached embeddings singleton for the configured provider."""
    if _override is not None:
        return _override
    settings = get_settings()
    model = (
        settings.openai_embedding_model
        if settings.embedding_provider == "openai"
        else settings.embedding_model
    )
    with _build_lock:
        return _build_embeddings(settings.embedding_provider, model, settings.embedding.batch_size)


def embedding_model_id() -> str:
    """Identifier stored in the Chroma collection metadata to detect model changes."""
    if _override is not None:
        return getattr(_override, "model_id", type(_override).__name__)
    return get_settings().active_embedding_model
