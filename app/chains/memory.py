"""Conversation memory: history formatting and follow-up condensing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.llm.factory import get_llm
from app.llm.prompts import CONDENSE_PROMPT

log = get_logger(__name__)
_MAX_TURN_CHARS = 1200


def _field(msg: Any, name: str) -> str:
    value = msg.get(name) if isinstance(msg, Mapping) else getattr(msg, name, "")
    return str(value or "")


def format_history(msgs: Sequence[Any] | None, turns: int | None = None) -> str:
    """Render the last ``turns`` user/assistant pairs as plain text.

    Accepts dicts or objects with ``role`` and ``content`` (e.g. ORM Message rows),
    oldest first. Long assistant answers are truncated to keep the condense prompt small.
    """
    if not msgs:
        return ""
    turns = get_settings().memory.history_turns if turns is None else turns
    if turns <= 0:
        return ""
    lines = []
    for msg in list(msgs)[-2 * turns :]:
        role = "Student" if _field(msg, "role") == "user" else "Assistant"
        content = _field(msg, "content").strip()
        if len(content) > _MAX_TURN_CHARS:
            content = content[:_MAX_TURN_CHARS] + " ..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def condense(question: str, history: str) -> str:
    """Rewrite a follow-up into a standalone question. Returns ``question`` unchanged
    when there is no history or the rewrite fails/looks unusable."""
    if not history.strip():
        return question
    chain = CONDENSE_PROMPT | get_llm() | StrOutputParser()
    try:
        rewritten = chain.invoke({"history": history, "question": question})
    except AppError:
        raise
    except Exception as exc:
        log.warning(
            "condense failed; using original question", extra={"fields": {"error": type(exc).__name__}}
        )
        return question
    rewritten = rewritten.strip().strip('"').strip()
    if rewritten.lower().startswith("standalone question:"):
        rewritten = rewritten.split(":", 1)[1].strip()
    if not rewritten or len(rewritten) > 4 * max(len(question), 100):
        return question
    log.info("condensed follow-up", extra={"fields": {"standalone": rewritten[:120]}})
    return rewritten
