"""Evaluation metrics: retrieval (Hit@k, MRR), refusal accuracy and LLM-as-judge scores."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.prompts import NOT_FOUND_MESSAGE


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk location."""

    source: str
    page: int


def _matches(hit: Hit, expected_source: str | None, expected_pages: Sequence[int]) -> bool:
    source_ok = not expected_source or hit.source.lower() == expected_source.lower()
    return source_ok and hit.page in set(expected_pages)


def first_relevant_rank(
    hits: Sequence[Hit], expected_source: str | None, expected_pages: Sequence[int]
) -> int:
    """1-based rank of the first retrieved chunk on an expected page (0 if none)."""
    for rank, hit in enumerate(hits, start=1):
        if _matches(hit, expected_source, expected_pages):
            return rank
    return 0


def hit_at_k(
    hits: Sequence[Hit], expected_source: str | None, expected_pages: Sequence[int], k: int
) -> float:
    """1.0 if an expected page appears in the top-k results, else 0.0."""
    rank = first_relevant_rank(hits[:k], expected_source, expected_pages)
    return 1.0 if rank else 0.0


def reciprocal_rank(hits: Sequence[Hit], expected_source: str | None, expected_pages: Sequence[int]) -> float:
    """1/rank of the first relevant chunk (0 if none). Averaged over queries this is MRR."""
    rank = first_relevant_rank(hits, expected_source, expected_pages)
    return 1.0 / rank if rank else 0.0


def is_refusal(answer: str) -> bool:
    """True if the answer is the system's 'not found' message."""
    norm = answer.strip().strip('"').lower()
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in norm


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean (0.0 for an empty list)."""
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------------------ LLM-as-judge

_FAITHFULNESS = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            # Claim-by-claim before scoring: small local judges (llama3.2:3b) answered "1" to
            # everything with a digit-only prompt, but discriminate well with this format.
            """You are checking whether an ANSWER is supported by the CONTEXT passages.
List the main claims of the ANSWER in one short line each and mark each SUPPORTED or NOT SUPPORTED by the CONTEXT.
Then give a final score from 1 to 5, where 5 means every claim is supported and 1 means most claims are not supported.
End with a line exactly like: SCORE: <number>

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}""",
        )
    ]
)

_RELEVANCE = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are grading a study assistant. Rate how directly and completely the ANSWER addresses the QUESTION.
5 = complete and on point, 4 = mostly complete, 3 = partial, 2 = barely relevant, 1 = irrelevant.
Reply with a single digit only.

QUESTION: {question}
REFERENCE ANSWER (may be empty): {reference}
ANSWER: {answer}
SCORE:""",
        )
    ]
)


def _parse_score(text: str) -> float | None:
    """The last ``SCORE: n`` in the reply, else the first digit 1-5."""
    labelled = re.findall(r"SCORE\s*[:=]\s*\**\s*([1-5])", text, re.IGNORECASE)
    if labelled:
        return float(labelled[-1])
    match = re.search(r"[1-5]", text)
    return float(match.group()) if match else None


def judge_faithfulness(llm: BaseChatModel, question: str, answer: str, context: str) -> float | None:
    """LLM-judged faithfulness of ``answer`` to ``context`` (1-5)."""
    chain = _FAITHFULNESS | llm | StrOutputParser()
    return _parse_score(chain.invoke({"question": question, "answer": answer, "context": context}))


def judge_relevance(llm: BaseChatModel, question: str, answer: str, reference: str = "") -> float | None:
    """LLM-judged relevance of ``answer`` to ``question`` (1-5)."""
    chain = _RELEVANCE | llm | StrOutputParser()
    return _parse_score(chain.invoke({"question": question, "answer": answer, "reference": reference}))
