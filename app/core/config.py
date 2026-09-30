"""Application settings.

Secrets and machine-specific values come from environment variables / ``.env``;
non-secret tunables come from ``config/settings.yaml``. Precedence (highest first):
init kwargs > environment variables > ``.env`` > ``settings.yaml`` > defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = PROJECT_ROOT / "config" / "settings.yaml"


def _yaml_path() -> Path:
    """Return the settings.yaml path (overridable with SETTINGS_YAML)."""
    return Path(os.environ.get("SETTINGS_YAML", DEFAULT_YAML))


def _resolve(path: Path) -> Path:
    """Resolve a relative path against the project root, not the CWD."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ChunkingSettings(BaseModel):
    """Text splitting parameters."""

    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(150, ge=0)
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]
    min_page_chars: int = Field(30, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalSettings(BaseModel):
    """Retriever parameters."""

    mode: Literal["similarity", "mmr", "hybrid"] = "mmr"
    top_k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=1)
    mmr_lambda: float = Field(0.6, ge=0.0, le=1.0)
    hybrid_weights: tuple[float, float] = (0.4, 0.6)
    score_threshold: float = 0.25
    use_reranker: bool = False
    rerank_top_n: int = Field(4, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class MemorySettings(BaseModel):
    """Conversation memory parameters."""

    history_turns: int = Field(4, ge=0)


class EmbeddingSettings(BaseModel):
    """Embedding batch parameters."""

    batch_size: int = Field(32, ge=1)


class Settings(BaseSettings):
    """All runtime configuration for the app, scripts and tests."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ---- LLM ----
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(700, ge=1)
    llm_timeout_s: float = Field(60.0, gt=0)

    # ---- Embeddings ----
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- Storage ----
    chroma_dir: Path = Path("./data/vectorstore")
    chroma_collection: str = "academic_docs"
    raw_dir: Path = Path("./data/raw")
    database_url: str = "sqlite:///./data/app.db"
    max_upload_mb: int = Field(50, ge=1)

    # ---- API / UI ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    backend_url: str = "http://127.0.0.1:8000"
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    log_level: str = "INFO"

    # ---- Tunables (settings.yaml) ----
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    memory: MemorySettings = MemorySettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chroma_dir", "raw_dir", mode="after")
    @classmethod
    def _absolute_dirs(cls, value: Path) -> Path:
        return _resolve(value)

    @field_validator("database_url", mode="after")
    @classmethod
    def _absolute_sqlite(cls, value: str) -> str:
        prefix = "sqlite:///"
        if value.startswith(prefix) and not value.startswith("sqlite:////"):
            rest = value[len(prefix) :]
            if rest and rest != ":memory:" and not Path(rest).is_absolute():
                return prefix + _resolve(Path(rest)).as_posix()
        return value

    @property
    def active_embedding_model(self) -> str:
        """Identifier of the embedding model currently configured (provider-qualified)."""
        if self.embedding_provider == "openai":
            return f"openai:{self.openai_embedding_model}"
        return f"local:{self.embedding_model}"

    @property
    def active_llm_model(self) -> str:
        """Name of the chat model currently configured."""
        return self.openai_model if self.llm_provider == "openai" else self.ollama_model

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_yaml_path())
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
