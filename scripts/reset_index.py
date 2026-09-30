"""Wipe the vector index.

Default: delete the Chroma collection, all rows of the ``documents`` table and the
stored PDFs (a clean slate; chat history is kept).
``--reindex``: delete only the Chroma collection and rebuild it from the stored
PDFs with the *current* embedding model (use after changing EMBEDDING_PROVIDER /
EMBEDDING_MODEL or chunking settings).

Usage:
    python scripts/reset_index.py [--yes]
    python scripts/reset_index.py --reindex
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.db import crud
from app.db.database import SessionLocal, init_db
from app.services.document_service import DocumentService
from app.vectorstore.store import reset_collection


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset the vector index.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild vectors from stored PDFs")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    args = parser.parse_args()
    s = get_settings()
    setup_logging(s.log_level)
    init_db()

    action = (
        "rebuild the vector index from stored PDFs"
        if args.reindex
        else ("DELETE the vector index, all document records and stored PDFs")
    )
    if not args.yes and input(f"This will {action}. Continue? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 1

    reset_collection()
    with SessionLocal() as db:
        records = list(crud.list_documents(db))
        if not args.reindex:
            for record in records:
                (s.raw_dir / f"{record.id}.pdf").unlink(missing_ok=True)
                db.delete(record)
            db.commit()
            print(f"Removed {len(records)} documents. Index is empty.")
            return 0

        svc, failures = DocumentService(db), 0
        for record in records:
            try:
                out = svc.reindex(record.id)
                print(f"[OK]   {out.filename}: {out.num_chunks} chunks")
            except AppError as exc:
                failures += 1
                print(f"[FAIL] {record.filename}: {exc.detail}")
    print(f"Reindexed with {s.active_embedding_model}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
