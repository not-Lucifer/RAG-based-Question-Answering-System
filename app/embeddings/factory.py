"""Embedding model factory (local sentence-transformers, Gemini or OpenAI)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
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


class RateLimitedEmbeddings(Embeddings):
    """Keeps an embeddings client under a per-minute request quota.

    Each text counts as one request. Calls wait until the sliding 60-second window has room,
    and a rejected call (quota exhausted) is retried after a pause instead of failing the upload.
    """

    def __init__(
        self,
        inner: Embeddings,
        per_minute: int,
        batch: int = 50,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._per_minute = per_minute
        self._batch = max(1, min(batch, per_minute))
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _acquire(self, n: int) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._sent and now - self._sent[0][0] >= 60:
                    self._sent.popleft()
                if sum(c for _, c in self._sent) + n <= self._per_minute:
                    self._sent.append((now, n))
                    return
                wait = 60 - (now - self._sent[0][0])
            self._sleep(max(wait, 1.0))

    def _call(self, fn: Callable[[], object]) -> object:
        for attempt in range(4):
            try:
                return fn()
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == 3:
                    raise
                log.warning("embedding quota reached; pausing", extra={"fields": {"attempt": attempt + 1}})
                self._sleep(40)
        raise AssertionError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            part = texts[i : i + self._batch]
            self._acquire(len(part))
            out.extend(self._call(lambda part=part: self._inner.embed_documents(part)))  # type: ignore[arg-type]
        return out

    def embed_query(self, text: str) -> list[float]:
        self._acquire(1)
        return self._call(lambda: self._inner.embed_query(text))  # type: ignore[return-value]


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
        if provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            if settings.gemini_api_key is None:
                raise EmbeddingError("GEMINI_API_KEY is not set; required for EMBEDDING_PROVIDER=gemini.")
            client = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=settings.gemini_api_key,
                output_dimensionality=settings.gemini_embedding_dim,
            )
            return RateLimitedEmbeddings(client, settings.gemini_embed_per_minute)
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
    model = {
        "openai": settings.openai_embedding_model,
        "gemini": settings.gemini_embedding_model,
    }.get(settings.embedding_provider, settings.embedding_model)
    with _build_lock:
        return _build_embeddings(settings.embedding_provider, model, settings.embedding.batch_size)


def embedding_model_id() -> str:
    """Identifier stored in the Chroma collection metadata to detect model changes."""
    if _override is not None:
        return getattr(_override, "model_id", type(_override).__name__)
    return get_settings().active_embedding_model
