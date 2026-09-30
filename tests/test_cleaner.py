"""Cleaner: header/footer removal, hyphenation, whitespace, short pages."""

from __future__ import annotations

from langchain_core.documents import Document

from app.ingestion.cleaner import clean_pages, clean_text
from app.ingestion.loader import load_pdf


def _docs(texts: list[str]) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": "x.pdf", "page": i + 1}) for i, t in enumerate(texts)
    ]


def test_repeated_header_and_page_numbers_removed() -> None:
    bodies = [
        "Body text about keys.",
        "Body text about joins.",
        "Body text on indexes.",
        "Body text on locks.",
    ]
    pages = [
        f"DBMS Notes - Unit 3\n{body} Enough words here.\nPage {i} of 4" for i, body in enumerate(bodies, 1)
    ]
    cleaned = clean_pages(_docs(pages))
    assert len(cleaned) == 4
    for doc in cleaned:
        assert "DBMS Notes" not in doc.page_content
        assert "Page" not in doc.page_content
        assert "Body text" in doc.page_content


def test_header_detection_needs_three_pages() -> None:
    cleaned = clean_pages(
        _docs(["Title line\nfirst page body text long enough", "Title line\nsecond body"]), 0
    )
    assert "Title line" in cleaned[0].page_content


def test_hyphenation_fixed() -> None:
    assert "algorithm" in clean_text("Dijkstra's algo-\nrithm finds paths")


def test_whitespace_normalised() -> None:
    out = clean_text("a   lot\tof    space\n\n\n\n\nnext para")
    assert out == "a lot of space\n\nnext para"


def test_short_pages_dropped() -> None:
    cleaned = clean_pages(_docs(["tiny", "This page has enough characters to be kept by the cleaner."]))
    assert [d.metadata["page"] for d in cleaned] == [2]


def test_real_pdf_headers_removed(make_pdf) -> None:
    body = "Paging divides memory into frames. The page table maps pages to frames."
    path = make_pdf([body] * 4, header="OS Class Notes (Semester 4)")
    cleaned = clean_pages(load_pdf(path))
    assert all("OS Class Notes" not in d.page_content and "Page 1 of" not in d.page_content for d in cleaned)
    assert all("page table" in d.page_content for d in cleaned)
