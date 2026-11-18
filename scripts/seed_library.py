"""Upload a folder of PDFs to a running API (used to seed the hosted demo after each restart).

The subject tag is the part of the file name before the first underscore
(``DBMS_Notes.pdf`` -> ``DBMS``). Files already indexed are skipped, and a failure never
stops the others, so a quota hiccup cannot take the demo down.

Usage:
    python scripts/seed_library.py --path data/samples --wait 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests


def wait_for_api(base: str, seconds: int) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base}/health", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the library through the API.")
    parser.add_argument("--path", type=Path, default=Path("data/samples"))
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for the API")
    parser.add_argument("--url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    base = args.url.rstrip("/")

    files = sorted(args.path.glob("*.pdf"))
    if not files:
        print(f"No PDFs in {args.path}; nothing to seed.")
        return 0
    if not wait_for_api(base, args.wait):
        print(f"API at {base} did not become ready within {args.wait}s; skipping seed.")
        return 1

    failures = 0
    for pdf in files:
        subject = pdf.stem.split("_")[0]
        try:
            with pdf.open("rb") as fh:
                res = requests.post(
                    f"{base}/documents/upload",
                    files={"file": (pdf.name, fh, "application/pdf")},
                    data={"subject": subject},
                    timeout=600,
                )
        except requests.RequestException as exc:
            failures += 1
            print(f"[FAIL] {pdf.name}: {exc}")
            continue
        if res.status_code == 201:
            body = res.json()
            print(f"[OK]   {pdf.name}: {body.get('num_pages')} pages, {body.get('num_chunks')} chunks")
        elif res.status_code == 409:
            print(f"[SKIP] {pdf.name}: already indexed")
        else:
            failures += 1
            print(f"[FAIL] {pdf.name}: HTTP {res.status_code} {res.text[:200]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
