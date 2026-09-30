"""Ask questions from the terminal (debugging the RAG chain without the API).

Usage:
    python scripts/ask_cli.py "What is 3NF?"
    python scripts/ask_cli.py --mode hybrid --subject DBMS          # interactive chat with memory
    LLM_PROVIDER=ollama python scripts/ask_cli.py "Explain paging"
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from app.chains.rag_chain import answer
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging


def show(question: str, history: list[dict], args: argparse.Namespace) -> None:
    filters = {"subject": args.subject} if args.subject else None
    result = answer(question, history, filters=filters, top_k=args.top_k, mode=args.mode)
    if result.standalone_question and result.standalone_question != question:
        print(f"(standalone: {result.standalone_question})")
    print(f"\n{result.answer}\n")
    for s in result.sources:
        print(f"  [{s.n}] {s.source} p.{s.page}  score={s.score}")
    print(f"  timings: {result.timings_ms}\n")
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the RAG system from the terminal.")
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--mode", choices=["similarity", "mmr", "hybrid"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--subject", default=None)
    args = parser.parse_args()
    s = get_settings()
    setup_logging("WARNING" if args.question else s.log_level)
    print(f"LLM: {s.llm_provider}/{s.active_llm_model} | embeddings: {s.active_embedding_model}")

    history: list[dict] = []
    try:
        if args.question:
            show(args.question, history, args)
            return 0
        print("Interactive mode - empty line or Ctrl+C to quit.")
        while question := input("? ").strip():
            show(question, history, args)
    except AppError as exc:
        print(f"Error [{exc.code}]: {exc.detail}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
