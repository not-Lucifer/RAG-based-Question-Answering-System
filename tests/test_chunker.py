"""Chunker: size, overlap, metadata."""

from __future__ import annotations

from langchain_core.documents import Document

from app.ingestion.chunker import DEFAULT_SUBJECT, split_documents

LONG = " ".join(f"Sentence number {i} explains an important academic concept in detail." for i in range(60))


def _pages() -> list[Document]:
    return [
        Document(page_content=LONG, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content=LONG, metadata={"source": "a.pdf", "page": 2}),
    ]


def test_chunks_respect_size() -> None:
    chunks = split_documents(_pages(), "d1", "DBMS", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 4
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = [c for c in split_documents(_pages(), "d1", "DBMS", 300, 80) if c.metadata["page"] == 1]
    overlaps = 0
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail_words = prev.page_content.split()[-3:]
        if " ".join(tail_words) in nxt.page_content:
            overlaps += 1
    assert overlaps >= len(chunks) - 2  # nearly every boundary shares text


def test_metadata_preserved_and_added() -> None:
    chunks = split_documents(_pages(), "doc-42", "DBMS", chunk_size=400, chunk_overlap=50)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.metadata["doc_id"] == "doc-42"
        assert c.metadata["subject"] == "DBMS"
        assert c.metadata["source"] == "a.pdf"
        assert c.metadata["page"] in (1, 2)
    assert {c.metadata["page"] for c in chunks} == {1, 2}


def test_blank_subject_defaults() -> None:
    chunks = split_documents(_pages(), "d", None)
    assert all(c.metadata["subject"] == DEFAULT_SUBJECT for c in chunks)


def test_default_settings_used() -> None:
    chunks = split_documents(_pages(), "d", "X")
    assert all(len(c.page_content) <= 1000 for c in chunks)
