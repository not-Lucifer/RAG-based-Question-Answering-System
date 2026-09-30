"""Evaluate retrieval (and optionally answers) on evaluation/qa_dataset.json.

For every chunk size a throwaway index is built from the sample PDFs in a temp
directory (your real index is never touched); every retrieval mode is then scored:

    Hit@k, MRR               - is the expected page retrieved, and how high?
    Refusal (retrieval)      - out-of-scope questions fall below score_threshold?
With --llm (needs a working LLM provider; costs API calls):
    Faithfulness, Relevance  - LLM-as-judge, 1-5
    Refusal (answer)         - the final answer is the "not found" message
    Latency                  - seconds per answered question

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --modes similarity mmr hybrid --chunk-sizes 500 1000 --k 5
    python scripts/run_eval.py --llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from langchain_core.documents import Document

from app.chains.rag_chain import answer, format_context, passes_threshold
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.ingestion.pipeline import ingest_file
from app.llm.factory import get_llm
from app.retrieval.hybrid import invalidate_bm25_cache
from app.retrieval.retriever import build_retriever
from app.vectorstore import store
from evaluation.metrics import (
    Hit,
    hit_at_k,
    is_refusal,
    judge_faithfulness,
    judge_relevance,
    mean,
    reciprocal_rank,
)

ROOT = _bootstrap.ROOT


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if not items:
        raise SystemExit(f"{path} has no items")
    return items


def build_index(samples: Path, workdir: Path, chunk_size: int) -> int:
    """Point settings at a fresh temp collection and ingest every sample PDF."""
    os.environ["CHROMA_DIR"] = str(workdir / f"chroma_{chunk_size}")
    os.environ["CHROMA_COLLECTION"] = f"eval_{chunk_size}"
    get_settings.cache_clear()
    store.reset_caches()
    invalidate_bm25_cache()
    overlap = min(get_settings().chunking.chunk_overlap, int(chunk_size * 0.15))
    total = 0
    for pdf in sorted(samples.glob("*.pdf")):
        res = ingest_file(
            pdf, pdf.stem, None, source_name=pdf.name, chunk_size=chunk_size, chunk_overlap=overlap
        )
        total += res.num_chunks
    return total


def eval_retrieval(items: list[dict[str, Any]], mode: str, k: int) -> dict[str, float]:
    retriever = build_retriever(mode=mode, top_k=k)
    hits, rrs, refusals = [], [], []
    for item in items:
        docs = retriever.invoke(item["question"])
        if item.get("in_scope", True):
            found = [Hit(d.metadata.get("source", ""), int(d.metadata.get("page", 0))) for d in docs]
            hits.append(hit_at_k(found, item.get("expected_source"), item["expected_pages"], k))
            rrs.append(reciprocal_rank(found, item.get("expected_source"), item["expected_pages"]))
        else:
            refusals.append(0.0 if passes_threshold(docs) else 1.0)
    return {
        "hit@k": mean(hits),
        "mrr": mean(rrs),
        "refusal_retrieval": mean(refusals) if refusals else float("nan"),
    }


def eval_answers(items: list[dict[str, Any]], mode: str, k: int) -> dict[str, float]:
    judge = get_llm()
    faith, rel, refusals, latency = [], [], [], []
    for item in items:
        started = time.perf_counter()
        result = answer(item["question"], mode=mode, top_k=k)
        latency.append(time.perf_counter() - started)
        if not item.get("in_scope", True):
            refusals.append(1.0 if is_refusal(result.answer) else 0.0)
            continue
        if not result.used_context:
            faith.append(1.0)
            rel.append(1.0)
            continue
        context = format_context(
            [
                Document(page_content=s.text, metadata={"source": s.source, "page": s.page})
                for s in result.sources
            ]
        )
        f = judge_faithfulness(judge, item["question"], result.answer, context)
        r = judge_relevance(judge, item["question"], result.answer, item.get("reference_answer", ""))
        faith += [f] if f is not None else []
        rel += [r] if r is not None else []
    return {
        "faithfulness": mean(faith),
        "relevance": mean(rel),
        "refusal_answer": mean(refusals) if refusals else float("nan"),
        "latency_s": mean(latency),
    }


def fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"  # NaN check


def to_markdown(rows: list[dict[str, Any]], with_llm: bool) -> str:
    cols = ["chunk_size", "mode", "hit@k", "mrr", "refusal_retrieval"]
    if with_llm:
        cols += ["faithfulness", "relevance", "refusal_answer", "latency_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append(
            "| " + " | ".join(fmt(row[c]) if isinstance(row[c], float) else str(row[c]) for c in cols) + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "qa_dataset.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--modes", nargs="+", default=["similarity", "mmr", "hybrid"])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="Also generate answers and run LLM judges")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()
    setup_logging("WARNING")

    items = load_dataset(args.dataset)
    in_scope = sum(1 for i in items if i.get("in_scope", True))
    print(
        f"Dataset: {len(items)} questions ({in_scope} in scope, "
        f"{len(items) - in_scope} out of scope), k={args.k}"
    )
    if len(items) < 20:
        print("Note: the blueprint targets 20-30 questions; add your own to evaluation/qa_dataset.json.")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rag-eval-", ignore_cleanup_errors=True) as tmp:
        try:
            for size in args.chunk_sizes:
                n_chunks = build_index(args.samples, Path(tmp), size)
                print(f"\nchunk_size={size}: indexed {n_chunks} chunks")
                for mode in args.modes:
                    row: dict[str, Any] = {
                        "chunk_size": size,
                        "mode": mode,
                        **eval_retrieval(items, mode, args.k),
                    }
                    if args.llm:
                        row.update(eval_answers(items, mode, args.k))
                    rows.append(row)
                    print(f"  {mode:<10} hit@{args.k}={fmt(row['hit@k'])} mrr={fmt(row['mrr'])}")
        except AppError as exc:
            print(f"Error [{exc.code}]: {exc.detail}", file=sys.stderr)
            return 1
        finally:
            store.reset_caches()

    table = to_markdown(rows, args.llm)
    print("\n" + table)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "dataset": str(args.dataset.name),
        "questions": len(items),
        "k": args.k,
        "embedding_model": get_settings().active_embedding_model,
        "score_threshold": get_settings().retrieval.score_threshold,
        "llm": get_settings().active_llm_model if args.llm else None,
    }
    (args.out / f"results-{stamp}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    (args.out / f"results-{stamp}.md").write_text(
        f"# Evaluation results ({stamp})\n\n```json\n{json.dumps(meta, indent=2)}\n```\n\n{table}\n"
    )
    print(f"\nSaved to {args.out / f'results-{stamp}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
