"""Evaluation metrics."""

from __future__ import annotations

from evaluation.metrics import (
    Hit,
    hit_at_k,
    is_refusal,
    judge_faithfulness,
    judge_relevance,
    mean,
    reciprocal_rank,
)

HITS = [Hit("a.pdf", 3), Hit("b.pdf", 2), Hit("a.pdf", 2)]


def test_hit_at_k() -> None:
    assert hit_at_k(HITS, "a.pdf", [2], k=3) == 1.0
    assert hit_at_k(HITS, "a.pdf", [2], k=2) == 0.0
    assert hit_at_k(HITS, None, [2], k=2) == 1.0  # any source


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(HITS, "a.pdf", [2]) == 1 / 3
    assert reciprocal_rank(HITS, "a.pdf", [9]) == 0.0
    assert mean([1.0, 0.5, 0.0]) == 0.5 and mean([]) == 0.0


def test_is_refusal() -> None:
    assert is_refusal("I couldn't find this in your uploaded material.")
    assert is_refusal('"I couldn\'t find this in your uploaded material"')
    assert not is_refusal("3NF removes transitive dependencies [1].")


def test_llm_judges_parse_scores(fake_llm) -> None:
    llm = fake_llm("Score: 4", "5")
    assert judge_faithfulness(llm, "q", "a", "ctx") == 4.0
    assert judge_relevance(llm, "q", "a") == 5.0
    assert judge_relevance(fake_llm("no idea"), "q", "a") is None
