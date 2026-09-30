"""Shared fixtures: isolated temp storage, deterministic fake embeddings and fake LLMs.

Every test runs offline: no model downloads, no API calls, no writes to ./data.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core import config as config_module
from app.core.config import get_settings
from app.db.database import dispose_engine, init_db
from app.embeddings import factory as emb_factory
from app.llm import factory as llm_factory
from app.retrieval.hybrid import invalidate_bm25_cache
from app.vectorstore import store

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or", "for", "what",
    "who", "how", "why", "which", "does", "do", "it", "its", "by", "with", "as", "be", "this", "that",
}  # fmt: skip


class FakeEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (lexical similarity, unit length)."""

    model_id = "fake:hash-bow-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]
        for token, n in Counter(tokens).items():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0 + math.log(n)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point all storage at a temp dir, ignore the developer's .env, install fake embeddings."""
    monkeypatch.setitem(config_module.Settings.model_config, "env_file", None)
    env = {
        "CHROMA_DIR": str(tmp_path / "chroma"),
        "CHROMA_COLLECTION": "test_docs",
        "RAW_DIR": str(tmp_path / "raw"),
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "",
        "EMBEDDING_PROVIDER": "local",
        "LOG_LEVEL": "WARNING",
        "WARMUP_ON_STARTUP": "false",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _reset_all()
    emb_factory.set_embeddings_override(FakeEmbeddings())
    init_db()
    yield tmp_path
    emb_factory.set_embeddings_override(None)
    llm_factory.set_llm_override(None)
    _reset_all()


def _reset_all() -> None:
    get_settings.cache_clear()
    store.reset_caches()
    invalidate_bm25_cache()
    dispose_engine()


class CountingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that also counts non-streaming calls in ``calls``."""

    calls: int = 0

    def _call(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        return super()._call(*args, **kwargs)


@pytest.fixture
def fake_llm() -> Callable[..., CountingFakeChatModel]:
    """Install a fake chat model returning ``responses`` in order (cycling)."""

    def _install(*responses: str) -> CountingFakeChatModel:
        llm = CountingFakeChatModel(responses=list(responses) or ["ok"])
        llm_factory.set_llm_override(llm)
        return llm

    return _install


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Create a PDF whose pages contain the given texts (``None`` = blank page)."""

    def _make(pages: list[str | None], name: str = "doc.pdf", header: str | None = None) -> Path:
        path = tmp_path / "pdfs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = pymupdf.open()
        for number, text in enumerate(pages, start=1):
            page = pdf.new_page(width=595, height=842)
            if header:
                page.insert_text((50, 40), header, fontsize=9)
            if text:
                page.insert_textbox(pymupdf.Rect(50, 60, 545, 790), text, fontsize=10)
            if header:
                page.insert_text((270, 815), f"Page {number} of {len(pages)}", fontsize=9)
        pdf.save(path)
        pdf.close()
        return path

    return _make


DBMS_PAGES = [
    "Normalization organises relations to reduce redundancy and remove insertion, update and deletion "
    "anomalies. Third normal form 3NF removes transitive dependency. BCNF requires every determinant "
    "to be a superkey.",
    "A B-tree is a balanced search tree. Advantages of B-trees: balanced height, logarithmic search, "
    "few disk reads because of high fan-out. B+ trees link their leaves for range queries.",
]
CN_PAGES = [
    "The TCP three-way handshake uses SYN, SYN-ACK and ACK segments to establish a connection.",
    "Dijkstra's algorithm computes shortest paths from a source. Time complexity of Dijkstra is "
    "O(V^2) with an array or O((V+E) log V) with a binary heap.",
]


@pytest.fixture
def indexed_docs(make_pdf: Callable[..., Path]) -> dict[str, Any]:
    """Ingest two small subject PDFs directly through the pipeline."""
    from app.ingestion.pipeline import ingest_file

    dbms = make_pdf(DBMS_PAGES, "dbms.pdf")
    cn = make_pdf(CN_PAGES, "cn.pdf")
    ingest_file(dbms, "doc-dbms", "DBMS")
    ingest_file(cn, "doc-cn", "CN")
    return {"dbms": dbms, "cn": cn}
