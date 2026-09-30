"""Evaluate retrieval (and optionally answers) on evaluation/qa_dataset.json.

For every chunk size a throwaway index is built in a temp directory from the READY
documents in your library (or from a folder with --samples); your real index is never
touched. Every retrieval mode is then scored:

    Hit@k, MRR               - is the expected page retrieved, and how high?
    Refusal (retrieval)      - out-of-scope questions fall below score_threshold?
With --llm (needs a working LLM provider; slow on Ollama, costs API calls on OpenAI):
    Faithfulness, Relevance  - LLM-as-judge, 1-5
    Refusal (answer)         - the final answer is the "not found" message
    Latency                  - seconds per answered question
    By default only for the configured retrieval mode at chunk size 1000; --llm-all for every row.

The dataset is validated first (sources exist, pages in range, unique ids).

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --modes similarity mmr hybrid --chunk-sizes 500 1000 --k 5
    python scripts/run_eval.py --llm
    python scripts/run_eval.py --samples data/samples --dataset my_sample_questions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from langchain_core.documents import Document

from app.chains.rag_chain import answer, format_context, passes_threshold
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.db import crud
from app.db.database import SessionLocal, init_db
from app.db.models import DocumentStatus
from app.ingestion.pipeline import ingest_file
from app.llm.factory import get_llm
from app.retrieval.hybrid import invalidate_bm25_cache
from app.retrieval.retriever import build_retriever
from app.vectorstore import store
from evaluation.dataset import load_dataset, unverified, validate_dataset
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


@dataclass(frozen=True)
class CorpusDoc:
    """A PDF to index for evaluation, cited under ``filename``."""

    path: Path
    filename: str
    doc_id: str
    subject: str | None
    pages: int


def library_corpus() -> list[CorpusDoc]:
    """READY documents from the app's library (stored PDFs + original filenames)."""
    init_db()
    raw_dir = get_settings().raw_dir
    with SessionLocal() as db:
        return [
            CorpusDoc(raw_dir / f"{d.id}.pdf", d.filename, d.id, d.subject, d.num_pages)
            for d in crud.list_documents(db)
            if d.status == DocumentStatus.READY.value and (raw_dir / f"{d.id}.pdf").exists()
        ]


def folder_corpus(folder: Path) -> list[CorpusDoc]:
    """Every PDF in a folder."""
    import pymupdf

    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        with pymupdf.open(pdf) as handle:
            docs.append(CorpusDoc(pdf, pdf.name, pdf.stem, None, handle.page_count))
    return docs


def build_index(corpus: list[CorpusDoc], workdir: Path, chunk_size: int) -> int:
    """Point settings at a fresh temp collection and ingest the corpus."""
    os.environ["CHROMA_DIR"] = str(workdir / f"chroma_{chunk_size}")
    os.environ["CHROMA_COLLECTION"] = f"eval_{chunk_size}"
    get_settings.cache_clear()
    store.reset_caches()
    invalidate_bm25_cache()
    overlap = min(get_settings().chunking.chunk_overlap, int(chunk_size * 0.15))
    total = 0
    for doc in corpus:
        res = ingest_file(
            doc.path,
            doc.doc_id,
            doc.subject,
            source_name=doc.filename,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        total += res.num_chunks
    return total


def eval_retrieval(items: list[dict[str, Any]], mode: str, k: int) -> dict[str, float]:
    retriever = build_retriever(mode=mode, top_k=k)
    hits, rrs, refusals, misses = [], [], [], []
    for item in items:
        docs = retriever.invoke(item["question"])
        if item.get("in_scope", True):
            found = [Hit(d.metadata.get("source", ""), int(d.metadata.get("page", 0))) for d in docs]
            hit = hit_at_k(found, item.get("expected_source"), item["expected_pages"], k)
            hits.append(hit)
            rrs.append(reciprocal_rank(found, item.get("expected_source"), item["expected_pages"]))
            if not hit:
                got = [f"{h.source[:14]} p{h.page}" for h in found[:3]]
                misses.append(f"{item['id']} expected p{item['expected_pages']}, got {got}")
        else:
            refused = not passes_threshold(docs)
            refusals.append(1.0 if refused else 0.0)
            if not refused:
                best = max(d.metadata.get("score", 0) for d in docs)
                misses.append(f"{item['id']} should be refused but passed the threshold (best={best})")
    return {
        "hit@k": mean(hits),
        "mrr": mean(rrs),
        "refusal_retrieval": mean(refusals) if refusals else float("nan"),
        "misses": misses,
    }


def eval_answers(items: list[dict[str, Any]], mode: str, k: int) -> dict[str, Any]:
    """Generate answers and score them; per-question details are kept for review."""
    judge = get_llm()
    faith, rel, refusals, latency, details = [], [], [], [], []
    for item in items:
        started = time.perf_counter()
        result = answer(item["question"], mode=mode, top_k=k)
        seconds = time.perf_counter() - started
        latency.append(seconds)
        record: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "answer": result.answer,
            "refused": is_refusal(result.answer),
            "cited": [f"{s.source} p{s.page}" for s in result.sources],
            "seconds": round(seconds, 1),
        }
        details.append(record)
        if not item.get("in_scope", True):
            refusals.append(1.0 if record["refused"] else 0.0)
            continue
        if not result.used_context:
            record.update(faithfulness=1.0, relevance=1.0)  # wrongly refused: worst score
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
        record.update(faithfulness=f, relevance=r)
        faith += [f] if f is not None else []
        rel += [r] if r is not None else []
    return {
        "faithfulness": mean(faith),
        "relevance": mean(rel),
        "refusal_answer": mean(refusals) if refusals else float("nan"),
        "latency_s": mean(latency),
        "answers": details,
    }


def fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"  # NaN check


def to_markdown(rows: list[dict[str, Any]], with_llm: bool) -> str:
    cols = ["chunk_size", "mode", "hit@k", "mrr", "refusal_retrieval"]
    if with_llm:
        cols += ["faithfulness", "relevance", "refusal_answer", "latency_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        cells = [row.get(c, "-") for c in cols]
        lines.append("| " + " | ".join(fmt(c) if isinstance(c, float) else str(c) for c in cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "qa_dataset.json")
    parser.add_argument("--samples", type=Path, default=None, help="Use PDFs in this folder, not the library")
    parser.add_argument("--modes", nargs="+", default=["similarity", "mmr", "hybrid"])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="Also generate answers and run LLM judges")
    parser.add_argument("--llm-all", action="store_true", help="Run the LLM pass for every row, not just one")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()
    setup_logging("WARNING")

    items = load_dataset(args.dataset)
    corpus = folder_corpus(args.samples) if args.samples else library_corpus()
    if not corpus:
        print("No documents to evaluate. Upload PDFs in the app (or pass --samples <folder>).")
        return 1
    print("Corpus: " + ", ".join(f"{d.filename} ({d.pages}p)" for d in corpus))
    problems = validate_dataset(items, {d.filename: d.pages for d in corpus})
    if problems:
        print("\nFix these dataset problems first:\n  - " + "\n  - ".join(problems))
        return 1
    pending = unverified(items)
    ai_checked = sum(1 for i in items if i.get("ai_checked"))
    if pending:
        print(
            f"Provenance: {len(items) - len(pending)} student-verified, {ai_checked} AI-checked. "
            f"Not yet verified by you: {len(pending)} "
            f"({', '.join(pending[:6])}{', ...' if len(pending) > 6 else ''})"
        )
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
                n_chunks = build_index(corpus, Path(tmp), size)
                print(f"\nchunk_size={size}: indexed {n_chunks} chunks")
                for mode in args.modes:
                    row: dict[str, Any] = {
                        "chunk_size": size,
                        "mode": mode,
                        **eval_retrieval(items, mode, args.k),
                    }
                    main_row = mode == get_settings().retrieval.mode and size == 1000
                    if args.llm and (args.llm_all or main_row):
                        print(f"  {mode:<10} generating + judging answers (slow)...")
                        row.update(eval_answers(items, mode, args.k))
                    rows.append(row)
                    print(f"  {mode:<10} hit@{args.k}={fmt(row['hit@k'])} mrr={fmt(row['mrr'])}")
                    for miss in row["misses"]:
                        print(f"             miss: {miss}")
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
        "unverified_questions": len(pending),
        "ai_checked_questions": ai_checked,
        "corpus": [d.filename for d in corpus],
        "k": args.k,
        "embedding_model": get_settings().active_embedding_model,
        "score_threshold": get_settings().retrieval.score_threshold,
        "llm": get_settings().active_llm_model if args.llm else None,
    }
    (args.out / f"results-{stamp}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / f"results-{stamp}.md").write_text(
        f"# Evaluation results ({stamp})\n\n```json\n{json.dumps(meta, indent=2)}\n```\n\n{table}\n",
        encoding="utf-8",
    )
    for row in rows:  # human-readable answers for review and report examples
        if "answers" not in row:
            continue
        lines = [f"# Answers: {row['mode']}, chunk_size {row['chunk_size']} ({stamp})\n"]
        for a in row["answers"]:
            scores = f"faithfulness={a.get('faithfulness', '-')} relevance={a.get('relevance', '-')}"
            lines.append(
                f"## {a['id']}: {a['question']}\n\n*{scores} · refused={a['refused']} · {a['seconds']}s · "
                f"cited: {', '.join(a['cited']) or 'none'}*\n\n{a['answer']}\n"
            )
        name = f"answers-{stamp}-{row['mode']}-{row['chunk_size']}.md"
        (args.out / name).write_text("\n".join(lines), encoding="utf-8")
        print(f"Answers saved to {args.out / name}")
    print(f"\nSaved to {args.out / f'results-{stamp}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
