"""Page text cleaning: headers/footers, page numbers, hyphenation, bullets, fragments, whitespace."""

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
# Bullet glyphs, incl. Wingdings/Symbol private-use characters and U+FFFD that PDF fonts
# often extract as; "o" and "-" are only treated as bullets when alone on a line.
_BULLET_CHARS = "•▪●◦■□➢►▶✓✔❖∙·*�"
_BULLET_ONLY = re.compile(rf"^(?:[{_BULLET_CHARS}-]|o|-)$")
_BULLET_LEAD = re.compile(rf"^[{_BULLET_CHARS}-]\s*")
_FRAGMENT_MAX = 25  # "There / is / a / Thin / Line" style extraction: one word or phrase per line
_SENTENCE_END = (".", ":", "?", "!", ";")
_LIST_START = re.compile(r"^(?:- |\d{1,2}[.)] |\(?[ivx]{1,4}\) |[a-z]\) )", re.IGNORECASE)


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
    # keys under 3 chars are bullets ("•", "o"), not headers; page numbers are handled separately
    return {key for key, n in counts.items() if len(key) >= 3 and n > len(pages) / 2}


def normalise_bullets(lines: list[str]) -> list[str]:
    """Turn bullet glyphs into "- " list markers, attaching lone bullets to the next line."""
    out: list[str] = []
    pending = False
    for line in lines:
        if _BULLET_ONLY.match(line):
            pending = True
            continue
        if not line:
            out.append(line)
            continue
        line = _BULLET_LEAD.sub("- ", line) if _BULLET_LEAD.match(line) else line
        if pending and not line.startswith("- "):
            line = "- " + line
        pending = False
        out.append(line)
    return out


def _is_fragment(line: str) -> bool:
    return 0 < len(line) <= _FRAGMENT_MAX and not line.endswith(_SENTENCE_END)


def join_fragments(lines: list[str]) -> list[str]:
    """Re-join text that PDF extraction split into one word/phrase per line.

    Only runs of at least two consecutive short fragments are joined (plus the line
    that completes them), so ordinary headings followed by a paragraph are kept.
    """
    out: list[str] = []
    joining = False
    for i, line in enumerate(lines):
        if joining and line and not line.startswith("- ") and not _LIST_START.match(line):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        joining = (
            _is_fragment(line)
            and bool(nxt)
            and not nxt.startswith("- ")
            and not _LIST_START.match(nxt)
            and (joining or _is_fragment(nxt))
        )
    return out


def clean_text(text: str, drop_keys: set[str] | frozenset[str] = frozenset()) -> str:
    """Clean a single page of text."""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [
        ln
        for ln in text.split("\n")
        if not _PAGE_NUMBER.match(ln) and (not ln.strip() or _line_key(ln) not in drop_keys)
    ]
    lines = [_SPACES.sub(" ", ln).strip() for ln in lines]
    text = _HYPHEN_BREAK.sub(r"\1\2", "\n".join(normalise_bullets(lines)))
    text = "\n".join(join_fragments(text.split("\n")))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def clean_pages(docs: list[Document], min_page_chars: int | None = None) -> list[Document]:
    """Clean page Documents and drop near-empty pages.

    Removes lines repeated on >50% of pages (running headers/footers), bare page
    numbers, joins words hyphenated across line breaks ("algo-\\nrithm" ->
    "algorithm"), turns bullet glyphs into "- " items, re-joins one-word-per-line
    fragments and normalises whitespace. Metadata is preserved.
    """
    min_chars = get_settings().chunking.min_page_chars if min_page_chars is None else min_page_chars
    drop = _repeated_lines([d.page_content for d in docs])
    cleaned: list[Document] = []
    for doc in docs:
        text = clean_text(doc.page_content, drop)
        if len(text) >= min_chars:
            cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return cleaned
