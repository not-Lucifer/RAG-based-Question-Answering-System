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
