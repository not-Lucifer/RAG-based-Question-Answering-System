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


def test_one_word_per_line_fragments_are_joined() -> None:
    raw = (
        "There\nis\na\nThin\nLine\nDifference\nbetween\nBiodata/Resume\nand\nCurriculum Vitae.\n\nNext para."
    )
    assert (
        clean_text(raw)
        == "There is a Thin Line Difference between Biodata/Resume and Curriculum Vitae.\n\nNext para."
    )


def test_heading_before_paragraph_is_kept() -> None:
    raw = "Reluctance\nReluctance is the resistance offered to the flow of magnetic flux in a circuit."
    assert clean_text(raw).split("\n")[0] == "Reluctance"


def test_fragment_run_completes_with_long_line() -> None:
    raw = "Panel\nmeans a\nselection\ncommittee\nthat\nis appointed for\ninterviewing the candidate. Panel may include three."
    assert clean_text(raw) == (
        "Panel means a selection committee that is appointed for interviewing the candidate. Panel may include three."
    )


def test_bullet_glyphs_normalised() -> None:
    raw = "INTERPERSONAL SKILLS\n\ufffd\nGood Communication Skills.\n\u25aa\nEasily negotiate with people.\n\uf0a8 This is a traditional interview.\no\nListen to the subject carefully"
    assert clean_text(raw).split("\n") == [
        "INTERPERSONAL SKILLS",
        "- Good Communication Skills.",
        "- Easily negotiate with people.",
        "- This is a traditional interview.",
        "- Listen to the subject carefully",
    ]


def test_short_bullet_items_stay_separate() -> None:
    raw = "\u2022\ncustomers\n\u2022\nemployees\n\u2022\ninvestors"
    assert clean_text(raw).split("\n") == ["- customers", "- employees", "- investors"]


def test_bullets_are_not_mistaken_for_headers() -> None:
    pages = [f"Topic {t} explained here in enough words.\n\u2022\nPoint about {t}" for t in "ABCD"]
    cleaned = clean_pages(_docs(pages), 0)
    assert all("- Point about" in d.page_content for d in cleaned)
