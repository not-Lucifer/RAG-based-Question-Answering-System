"""Loading and validating evaluation/qa_dataset.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED = ("id", "question", "expected_source", "expected_pages", "reference_answer", "in_scope")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Return the ``items`` list (accepts a bare list too)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} contains no questions")
    return items


def validate_dataset(items: list[dict[str, Any]], page_counts: dict[str, int] | None = None) -> list[str]:
    """Return a list of problems (empty when the dataset is usable).

    Args:
        items: Dataset entries.
        page_counts: ``{filename: number_of_pages}`` of the corpus; when given, sources
            and page numbers are checked against it (case-insensitive filenames).
    """
    problems: list[str] = []
    seen: set[str] = set()
    known = {name.lower(): pages for name, pages in (page_counts or {}).items()}
    for n, item in enumerate(items, start=1):
        tag = f"item {n} ({item.get('id', '?')})"
        missing = [key for key in REQUIRED if key not in item]
        if missing:
            problems.append(f"{tag}: missing fields {missing}")
            continue
        if item["id"] in seen:
            problems.append(f"{tag}: duplicate id")
        seen.add(item["id"])
        if not str(item["question"]).strip():
            problems.append(f"{tag}: empty question")
        pages = item["expected_pages"]
        if item["in_scope"]:
            if not item["expected_source"] or not pages:
                problems.append(f"{tag}: in-scope questions need expected_source and expected_pages")
                continue
            if not all(isinstance(p, int) and p >= 1 for p in pages):
                problems.append(f"{tag}: expected_pages must be 1-based integers")
            if known:
                total = known.get(str(item["expected_source"]).lower())
                if total is None:
                    problems.append(f"{tag}: '{item['expected_source']}' is not in the corpus")
                elif any(isinstance(p, int) and p > total for p in pages):
                    problems.append(f"{tag}: page beyond end of '{item['expected_source']}' ({total} pages)")
        elif item["expected_source"] or pages:
            problems.append(f"{tag}: out-of-scope questions must have expected_source null and no pages")
    return problems


def unverified(items: list[dict[str, Any]]) -> list[str]:
    """Ids of items explicitly marked ``"verified": false`` (AI-drafted, not yet reviewed)."""
    return [str(i.get("id")) for i in items if i.get("verified") is False]
