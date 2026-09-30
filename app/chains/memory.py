"""Conversation memory: history formatting and follow-up condensing."""

from __future__ import annotations

import re
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


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# Words that point back at earlier turns ("explain ITS advantages", "what about THEM").
_REFERENCES = {
    "it", "its", "it's", "this", "that", "these", "those", "they", "them", "their", "theirs",
    "he", "she", "him", "her", "his", "hers", "above", "previous", "same", "former", "latter",
    "one", "ones", "more", "else", "again", "another", "other", "there", "such", "also",
    "last", "first", "second", "third", "fourth", "fifth", "next", "earlier", "mentioned",
    "point", "points", "part", "step", "steps", "option", "options",
}  # fmt: skip
_FOLLOW_UP_STARTS = (
    "and ", "but ", "so ", "then ", "also ", "what about", "how about", "why", "how so",
    "elaborate", "continue", "example", "give an example", "in short", "summarise", "summarize",
)  # fmt: skip
# Words that carry no topic, ignored when checking that a rewrite kept the question's subject.
_NON_TOPIC = _REFERENCES | {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for", "and", "or",
    "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did", "can",
    "could", "should", "would", "will", "i", "me", "my", "we", "you", "your", "with", "about",
    "explain", "describe", "define", "tell", "give", "list", "mention", "write", "state", "briefly",
    "detail", "details", "please", "some", "any", "all", "much", "many", "difference", "between",
}  # fmt: skip


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def needs_condensing(question: str) -> bool:
    """True if the question refers back to the conversation and must be rewritten.

    A complete question such as "What is a resume?" is used as-is, even mid-conversation.
    Small local models tend to 'rewrite' such questions into the previous topic.
    """
    q = question.strip().lower()
    return q.startswith(_FOLLOW_UP_STARTS) or any(w in _REFERENCES for w in _words(q))


def keeps_topic(question: str, rewritten: str) -> bool:
    """True if the rewrite still contains most topic words of the original question."""
    topic = {w for w in _words(question) if w not in _NON_TOPIC and len(w) > 1}
    if not topic:
        return True  # e.g. "why?" - nothing to check
    target = rewritten.lower()
    kept = sum(1 for w in topic if w[:5] in target)  # prefix match tolerates plurals/inflections
    return kept * 2 >= len(topic)


def extract_question(output: str) -> str:
    """Pull the rewritten question out of a chatty model reply.

    Handles preambles ("Here is the rewritten question:"), labels
    ("Standalone question: ...") and surrounding quotes/markdown.
    """
    lines = [ln.strip().strip("*_`\"'").strip() for ln in output.strip().splitlines()]
    cleaned = []
    for line in lines:
        if ":" in line and line.split(":", 1)[0].lower().strip() in {
            "standalone question", "rewritten question", "question", "follow-up", "answer",
        }:  # fmt: skip
            line = line.split(":", 1)[1].strip().strip("\"'")
        if line and not line.endswith(":"):  # drop "Here is the rewritten question:" lines
            cleaned.append(line)
    questions = [ln for ln in cleaned if ln.endswith("?")]
    return (questions or cleaned or [""])[0]


def condense(question: str, history: str) -> str:
    """Rewrite a follow-up into a standalone question.

    Returns ``question`` unchanged when there is no history, when the question is
    already standalone, or when the rewrite fails / drops the question's topic.
    """
    if not history.strip() or not needs_condensing(question):
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
    rewritten = extract_question(rewritten)
    if not rewritten or len(rewritten) > 4 * max(len(question), 100):
        return question
    if not keeps_topic(question, rewritten):
        log.warning("rewrite dropped the question's topic; using original", extra={"fields": {"q": question}})
        return question
    log.info("condensed follow-up", extra={"fields": {"standalone": rewritten[:120]}})
    return rewritten
