"""PDF loader: page numbers, metadata, empty/invalid PDFs."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import EmptyPDFError, InvalidFileError
from app.ingestion.loader import load_pdf


def test_one_document_per_page_with_1_based_pages(make_pdf) -> None:
    path = make_pdf(["alpha page text", "beta page text", "gamma page text"], "notes.pdf")
    docs = load_pdf(path)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert all(d.metadata["source"] == "notes.pdf" for d in docs)
    assert "beta" in docs[1].page_content


def test_source_name_override(make_pdf) -> None:
    docs = load_pdf(make_pdf(["some text here"]), source_name="DBMS_Unit3.pdf")
    assert docs[0].metadata["source"] == "DBMS_Unit3.pdf"


def test_blank_pages_kept_for_numbering(make_pdf) -> None:
    docs = load_pdf(make_pdf(["first", None, "third"]))
    assert len(docs) == 3 and docs[2].metadata["page"] == 3


def test_empty_pdf_raises(make_pdf) -> None:
    with pytest.raises(EmptyPDFError):
        load_pdf(make_pdf([None, None]))


def test_non_pdf_raises_invalid_file(tmp_path: Path) -> None:
    bogus = tmp_path / "fake.pdf"
    bogus.write_bytes(b"this is not a pdf at all")
    with pytest.raises(InvalidFileError):
        load_pdf(bogus)
