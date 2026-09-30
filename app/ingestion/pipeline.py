"""End-to-end ingestion: load -> clean -> chunk -> embed + store."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import EmptyPDFError
from app.core.logging import get_logger
from app.ingestion.chunker import split_documents
from app.ingestion.cleaner import clean_pages
from app.ingestion.loader import load_pdf

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting one file."""

    num_pages: int
    num_chunks: int


def ingest_file(
    path: Path,
    doc_id: str,
    subject: str | None,
    source_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Ingest one PDF into the vector store.

    Any existing chunks of ``doc_id`` are removed first, so re-ingesting (reindex)
    never leaves stale chunks behind.

    Raises:
        InvalidFileError / EmptyPDFError: unreadable or text-less PDF.
        EmbeddingError / VectorStoreError / IndexMismatchError: storage failures.
    """
    from app.vectorstore.store import add_chunks, delete_doc  # local: keeps Phase-1 tools light

    started = time.perf_counter()
    pages = load_pdf(path, source_name=source_name)
    cleaned = clean_pages(pages)
    if not cleaned:
        raise EmptyPDFError()
    chunks = split_documents(cleaned, doc_id, subject, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    delete_doc(doc_id)
    add_chunks(chunks)
    log.info(
        "ingested document",
        extra={
            "fields": {
                "doc_id": doc_id,
                "pages": len(pages),
                "chunks": len(chunks),
                "ms": int((time.perf_counter() - started) * 1000),
            }
        },
    )
    return IngestResult(num_pages=len(pages), num_chunks=len(chunks))
