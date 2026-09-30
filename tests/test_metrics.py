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


def test_validate_dataset_catches_problems() -> None:
    from evaluation.dataset import unverified, validate_dataset

    good = {"id": "a", "question": "q?", "expected_source": "x.pdf", "expected_pages": [2],
            "reference_answer": "r", "in_scope": True, "verified": False}  # fmt: skip
    oos = {"id": "o", "question": "q?", "expected_source": None, "expected_pages": [],
           "reference_answer": "", "in_scope": False}  # fmt: skip
    assert validate_dataset([good, oos], {"X.pdf": 3}) == []
    assert unverified([good, oos]) == ["a"]
    bad = [
        dict(good),  # duplicate id "a"
        {**good, "id": "b", "expected_pages": [9]},
        {**good, "id": "c", "expected_source": "y.pdf"},
        {**oos, "id": "d", "expected_pages": [1]},
        {"id": "e"},
    ]
    problems = validate_dataset([good, *bad], {"x.pdf": 3})
    text = " | ".join(problems)
    for expected in ("duplicate id", "beyond end", "not in the corpus", "out-of-scope", "missing fields"):
        assert expected in text


def test_shipped_datasets_are_well_formed() -> None:
    from pathlib import Path

    from evaluation.dataset import load_dataset, validate_dataset

    root = Path(__file__).resolve().parents[1] / "evaluation"
    for name in ("qa_dataset.json", "sample_dataset.json"):
        assert validate_dataset(load_dataset(root / name)) == [], name


def test_score_parsing_prefers_final_score_line(fake_llm) -> None:
    reply = "1. Claim A - SUPPORTED\n2. Claim B - SUPPORTED\n3. Claim C - NOT SUPPORTED\nSCORE: 4"
    assert judge_faithfulness(fake_llm(reply), "q", "a", "ctx") == 4.0
    assert judge_faithfulness(fake_llm("**SCORE:** 5"), "q", "a", "ctx") == 5.0
