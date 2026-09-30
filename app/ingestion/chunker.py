"""Chunking with RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

DEFAULT_SUBJECT = "General"


def split_documents(
    docs: list[Document],
    doc_id: str,
    subject: str | None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split cleaned pages into overlapping chunks.

    Every chunk keeps its page's metadata (``source``, ``page``) and gains
    ``doc_id``, ``subject`` and a document-wide ``chunk_index``. Chunks never span
    pages, so page citations stay exact.

    Args:
        docs: Cleaned page Documents.
        doc_id: Owning document id.
        subject: Subject tag; blank means ``"General"`` (Chroma metadata cannot be null).
        chunk_size: Override ``chunking.chunk_size`` (used by evaluation).
        chunk_overlap: Override ``chunking.chunk_overlap``.
    """
    cfg = get_settings().chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap if chunk_overlap is None else chunk_overlap,
        separators=cfg.separators,
        keep_separator=True,
        strip_whitespace=True,
    )
    subject = (subject or "").strip() or DEFAULT_SUBJECT
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update({"doc_id": doc_id, "subject": subject, "chunk_index": index})
    return chunks
