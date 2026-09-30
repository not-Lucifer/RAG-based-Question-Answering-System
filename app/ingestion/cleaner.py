"""Page text cleaning: headers/footers, page numbers, hyphenation, whitespace."""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.documents import Document

from app.core.config import get_settings

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n\s*(\w)")
_SPACES = re.compile(r"[ \t\f\v ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_HEADER_LEN = 120
_MIN_PAGES_FOR_HEADER_DETECTION = 3


def _line_key(line: str) -> str:
    """Normalise a line so 'Unit 3 - Page 4' and 'Unit 3 - Page 5' compare equal."""
    return re.sub(r"\d+", "#", _SPACES.sub(" ", line).strip().lower())


def _repeated_lines(pages: list[str]) -> set[str]:
    """Short lines that appear on more than half of the pages (headers/footers)."""
    if len(pages) < _MIN_PAGES_FOR_HEADER_DETECTION:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        keys = {_line_key(ln) for ln in text.splitlines() if ln.strip() and len(ln) <= _MAX_HEADER_LEN}
        counts.update(keys)
    return {key for key, n in counts.items() if key and n > len(pages) / 2}


def clean_text(text: str, drop_keys: set[str] | frozenset[str] = frozenset()) -> str:
    """Clean a single page of text."""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [
        ln
        for ln in text.split("\n")
        if not _PAGE_NUMBER.match(ln) and (not ln.strip() or _line_key(ln) not in drop_keys)
    ]
    text = "\n".join(_SPACES.sub(" ", ln).strip() for ln in lines)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def clean_pages(docs: list[Document], min_page_chars: int | None = None) -> list[Document]:
    """Clean page Documents and drop near-empty pages.

    Removes lines repeated on >50% of pages (running headers/footers), bare page
    numbers, joins words hyphenated across line breaks ("algo-\\nrithm" ->
    "algorithm") and normalises whitespace. Metadata is preserved.
    """
    min_chars = get_settings().chunking.min_page_chars if min_page_chars is None else min_page_chars
    drop = _repeated_lines([d.page_content for d in docs])
    cleaned: list[Document] = []
    for doc in docs:
        text = clean_text(doc.page_content, drop)
        if len(text) >= min_chars:
            cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return cleaned
