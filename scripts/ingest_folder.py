"""Bulk-ingest a folder of PDFs.

Registers each file in SQLite (dedupe by hash) and indexes it in Chroma, exactly
like an upload through the API. ``--dry-run`` only loads/cleans/chunks and prints
what would be indexed (no embeddings, no writes).

Usage:
    python scripts/ingest_folder.py --path data/samples --subject DBMS
    python scripts/ingest_folder.py --path data/samples --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.config import get_settings
from app.core.exceptions import AppError, DuplicateDocumentError
from app.core.logging import setup_logging
from app.db.database import SessionLocal, init_db
from app.ingestion.chunker import split_documents
from app.ingestion.cleaner import clean_pages
from app.ingestion.loader import load_pdf
from app.services.document_service import DocumentService


def _print_samples(chunks: list, n: int = 2) -> None:
    for chunk in chunks[:n]:
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"    - {chunk.metadata}\n      {preview}...")


def dry_run(files: list[Path], subject: str | None) -> None:
    for path in files:
        try:
            pages = load_pdf(path)
            chunks = split_documents(clean_pages(pages), "dry-run", subject)
        except AppError as exc:
            print(f"[FAIL] {path.name}: {exc.detail}")
            continue
        print(f"[OK]   {path.name}: {len(pages)} pages, {len(chunks)} chunks")
        _print_samples(chunks)


def ingest(files: list[Path], subject: str | None) -> int:
    init_db()
    failures = 0
    with SessionLocal() as db:
        svc = DocumentService(db)
        for path in files:
            try:
                res = svc.upload(path.name, path.read_bytes(), subject)
                print(f"[OK]   {path.name}: {res.num_pages} pages, {res.num_chunks} chunks (id={res.doc_id})")
            except DuplicateDocumentError:
                print(f"[SKIP] {path.name}: already indexed")
            except AppError as exc:
                failures += 1
                print(f"[FAIL] {path.name}: {exc.detail}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-ingest a folder of PDFs.")
    parser.add_argument("--path", type=Path, default=Path("data/samples"))
    parser.add_argument("--subject", default=None, help="Subject tag for all files (default: General)")
    parser.add_argument("--dry-run", action="store_true", help="Only load/clean/chunk and print")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)

    folder = args.path if args.path.is_absolute() else (Path.cwd() / args.path)
    files = sorted(folder.glob("*.pdf"))
    if not files:
        print(f"No PDFs found in {folder}. Tip: `python scripts/make_samples.py` creates sample notes.")
        return 1
    if args.dry_run:
        dry_run(files, args.subject)
        return 0
    return 1 if ingest(files, args.subject) else 0


if __name__ == "__main__":
    sys.exit(main())
