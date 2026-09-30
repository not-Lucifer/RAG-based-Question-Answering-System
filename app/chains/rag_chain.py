"""The RAG chain: condense -> retrieve -> (rerank) -> threshold -> prompt -> LLM."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.chains.memory import condense, format_history
from app.core.config import get_settings
from app.core.exceptions import AppError, LLMUnavailableError
from app.core.logging import get_logger
from app.llm.factory import get_llm
from app.llm.prompts import NOT_FOUND_MESSAGE, QA_PROMPT
from app.retrieval.retriever import Mode, build_retriever

log = get_logger(__name__)
SNIPPET_CHARS = 300


@dataclass
class Source:
    """A numbered context passage, as cited by ``[n]`` in the answer."""

    n: int
    source: str
    page: int
    snippet: str
    score: float | None
    doc_id: str
    chunk_index: int
    subject: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (for JSON storage)."""
        return asdict(self)


@dataclass
class RAGResult:
    """Answer plus the sources in citation order."""

    answer: str
    sources: list[Source]
    used_context: bool
    standalone_question: str = ""
    timings_ms: dict[str, int] = field(default_factory=dict)


def best_score(docs: Sequence[Document]) -> float | None:
    """Highest cosine score among retrieved documents (``None`` if unscored)."""
    scores = [d.metadata["score"] for d in docs if isinstance(d.metadata.get("score"), int | float)]
    return max(scores) if scores else None


def passes_threshold(docs: Sequence[Document], threshold: float | None = None) -> bool:
    """True when at least one chunk is relevant enough to answer from."""
    if not docs:
        return False
    threshold = get_settings().retrieval.score_threshold if threshold is None else threshold
    top = best_score(docs)
    return top is None or top >= threshold


def format_context(docs: Sequence[Document]) -> str:
    """Numbered context blocks: ``[1] (file.pdf, p.12)\\n<chunk text>``."""
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def to_sources(docs: Sequence[Document]) -> list[Source]:
    """Convert context documents to citation-ordered sources."""
    out = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        text = d.page_content
        out.append(
            Source(
                n=i,
                source=str(m.get("source", "unknown")),
                page=int(m.get("page", 0)),
                snippet=text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else ""),
                score=m.get("score"),
                doc_id=str(m.get("doc_id", "")),
                chunk_index=int(m.get("chunk_index", -1)),
                subject=str(m.get("subject", "")),
                text=text,
            )
        )
    return out


def _maybe_rerank(inputs: dict[str, Any]) -> list[Document]:
    docs = inputs["docs"]
    cfg = get_settings().retrieval
    if cfg.use_reranker and docs:
        from app.retrieval.reranker import rerank

        return rerank(inputs["question"], docs, cfg.rerank_top_n)
    return docs


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel | None = None) -> Runnable:
    """LCEL chain: ``{"question"} -> {"answer", "docs", "used_context"}``.

    retriever -> optional rerank -> score gate -> format context -> prompt -> LLM -> StrOutputParser.
    When nothing passes the score threshold the LLM is not called.
    """
    llm = llm or get_llm()
    generate = QA_PROMPT | llm | StrOutputParser()

    def _answer_or_refuse(inputs: dict[str, Any]) -> dict[str, Any]:
        docs = inputs["docs"]
        if not passes_threshold(docs):
            return {"answer": NOT_FOUND_MESSAGE, "docs": [], "used_context": False}
        answer = generate.invoke({"context": format_context(docs), "question": inputs["question"]})
        return {"answer": answer.strip(), "docs": docs, "used_context": True}

    return (
        RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
        | RunnablePassthrough.assign(docs=RunnableLambda(_maybe_rerank))
        | RunnableLambda(_answer_or_refuse)
    )


def _standalone(question: str, history: Sequence[Any] | None) -> str:
    return condense(question, format_history(history))


def answer(
    question: str,
    history: Sequence[Any] | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    mode: Mode | None = None,
) -> RAGResult:
    """Answer a question from the indexed material.

    Args:
        question: The user's (possibly follow-up) question.
        history: Earlier messages (dicts/objects with ``role``/``content``), oldest first.
        filters: ``{"subject": ..., "doc_ids": [...]}``.
        top_k: Number of chunks to retrieve (defaults to settings).
        mode: Retrieval mode override (similarity | mmr | hybrid).

    Raises:
        LLMUnavailableError, EmbeddingError, IndexMismatchError, VectorStoreError.
    """
    t0 = time.perf_counter()
    standalone = _standalone(question, history)
    t1 = time.perf_counter()
    chain = build_rag_chain(build_retriever(mode=mode, top_k=top_k, filters=filters))
    try:
        out = chain.invoke({"question": standalone})
    except AppError:
        raise
    except Exception as exc:
        log.error("rag chain failed", extra={"fields": {"error": type(exc).__name__}})
        raise LLMUnavailableError(
            "The language model request failed. Check the provider and try again."
        ) from exc
    t2 = time.perf_counter()
    timings = {"condense_ms": int((t1 - t0) * 1000), "retrieve_generate_ms": int((t2 - t1) * 1000)}
    log.info(
        "answered",
        extra={"fields": {"used_context": out["used_context"], "chunks": len(out["docs"]), **timings}},
    )
    return RAGResult(
        answer=out["answer"],
        sources=to_sources(out["docs"]),
        used_context=out["used_context"],
        standalone_question=standalone,
        timings_ms=timings,
    )


def stream_answer(
    question: str,
    history: Sequence[Any] | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    mode: Mode | None = None,
) -> Iterator[tuple[str, Any]]:
    """Streaming variant of :func:`answer`.

    Yields ``("token", str)`` events while the LLM generates, then a single
    ``("result", RAGResult)`` event with the full answer and sources.
    """
    standalone = _standalone(question, history)
    docs = build_retriever(mode=mode, top_k=top_k, filters=filters).invoke(standalone)
    docs = _maybe_rerank({"docs": docs, "question": standalone})
    if not passes_threshold(docs):
        yield "token", NOT_FOUND_MESSAGE
        yield "result", RAGResult(NOT_FOUND_MESSAGE, [], False, standalone)
        return
    chain = QA_PROMPT | get_llm(streaming=True) | StrOutputParser()
    parts: list[str] = []
    try:
        for token in chain.stream({"context": format_context(docs), "question": standalone}):
            parts.append(token)
            yield "token", token
    except AppError:
        raise
    except Exception as exc:
        log.error("streaming failed", extra={"fields": {"error": type(exc).__name__}})
        raise LLMUnavailableError(
            "The language model request failed. Check the provider and try again."
        ) from exc
    yield "result", RAGResult("".join(parts).strip(), to_sources(docs), True, standalone)
