"""Settings loading and secret handling."""

from __future__ import annotations

import logging

import pytest

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.logging import RedactingFormatter, redact


def test_yaml_tunables_loaded() -> None:
    s = get_settings()
    assert s.chunking.chunk_size == 1000
    assert s.chunking.chunk_overlap == 150
    assert s.retrieval.mode == "mmr"
    assert s.retrieval.hybrid_weights == (0.4, 0.6)
    assert s.memory.history_turns == 4
    assert s.embedding.batch_size == 32


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL__TOP_K", "7")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()
    s = get_settings()
    assert s.retrieval.top_k == 7
    assert s.llm_provider == "ollama"
    assert s.active_llm_model == s.ollama_model


def test_blank_api_key_is_none() -> None:
    assert get_settings().openai_api_key is None


def test_relative_paths_resolve_against_project_root() -> None:
    s = Settings(chroma_dir="./data/vs", database_url="sqlite:///./data/x.db")
    assert s.chroma_dir == (PROJECT_ROOT / "data" / "vs").resolve()
    assert s.database_url.endswith("/data/x.db") and PROJECT_ROOT.as_posix() in s.database_url


def test_secret_never_printed() -> None:
    s = Settings(openai_api_key="sk-test-THISISSECRET123456")
    assert "THISISSECRET" not in repr(s)
    assert "THISISSECRET" not in str(s.model_dump())
    assert s.openai_api_key.get_secret_value().endswith("123456")


def test_log_redaction() -> None:
    assert "sk-***" in redact("key sk-abcdefghijklmnop123")
    assert "SECRET" not in redact("Authorization: Bearer SECRET")
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "api_key=sk-abcdefghijk12345", None, None)
    assert "abcdefghijk" not in RedactingFormatter("%(message)s").format(record)


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(chunking={"chunk_size": 100, "chunk_overlap": 100})


def test_gemini_settings_and_secret() -> None:
    s = Settings(
        llm_provider="gemini", embedding_provider="gemini", gemini_api_key="AIzaFAKEKEY0123456789abcdef"
    )
    assert s.active_llm_model == s.gemini_model
    assert s.active_embedding_model == f"gemini:{s.gemini_embedding_model}:{s.gemini_embedding_dim}"
    assert "AIzaFAKEKEY" not in repr(s)
    assert Settings(gemini_api_key="  ").gemini_api_key is None
    assert "AIza***" in redact("key=AIzaFAKEKEY0123456789abcdef")
    assert "AQ.***" in redact("x AQ.Ab8RN6FAKEKEY0123456789abcdef y")


def test_rate_limited_embeddings_wait_for_quota() -> None:
    from langchain_core.embeddings import FakeEmbeddings

    from app.embeddings.factory import RateLimitedEmbeddings

    now = [0.0]
    pauses: list[float] = []

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        now[0] += seconds

    emb = RateLimitedEmbeddings(
        FakeEmbeddings(size=4), per_minute=10, batch=5, clock=lambda: now[0], sleep=sleep
    )
    assert len(emb.embed_documents([f"t{i}" for i in range(10)])) == 10
    assert pauses == []  # exactly the quota: no waiting
    emb.embed_query("one more")
    assert pauses and now[0] >= 60  # the window had to expire first


def test_rate_limited_embeddings_retry_on_quota_error() -> None:
    from app.embeddings.factory import RateLimitedEmbeddings

    class Flaky:
        calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return [[0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    pauses: list[float] = []
    emb = RateLimitedEmbeddings(Flaky(), per_minute=100, sleep=pauses.append)  # type: ignore[arg-type]
    assert emb.embed_documents(["a", "b"]) == [[0.0], [0.0]]
    assert pauses == [40]
