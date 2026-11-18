"""Publish the current commit to a Hugging Face Space (Docker SDK).

A Space needs YAML metadata at the top of README.md, which would look odd on GitHub, so this
builds a throwaway copy of HEAD with the Space card prepended and force-pushes only that copy.
Your repository and working tree are not modified.

Usage:
    python scripts/push_to_space.py --space your-username/academic-rag-qa

Git asks for your Hugging Face username and an access token (with write permission) as the
password. Create the Space first (New Space, SDK: Docker, Blank) and add GEMINI_API_KEY under
Settings -> Variables and secrets -> New secret.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "deploy" / "huggingface" / "space_card.md"


def run(*cmd: str, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push HEAD to a Hugging Face Space.")
    parser.add_argument("--space", required=True, help="owner/name of the Space")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        archive = work / "head.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
        out = work / "space"
        out.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(out, filter="data")
        readme = out / "README.md"
        readme.write_text(
            CARD.read_text(encoding="utf-8") + readme.read_text(encoding="utf-8"), encoding="utf-8"
        )
        run("git", "init", "-q", "-b", "main", cwd=out)
        run("git", "add", "-A", cwd=out)
        run("git", "-c", "user.name=space-deploy", "-c", "user.email=space@localhost",
            "commit", "-q", "-m", "Deploy to Hugging Face Space", cwd=out)  # fmt: skip
        run("git", "push", "--force", f"https://huggingface.co/spaces/{args.space}", "main:main", cwd=out)
    print(f"Pushed. Build logs: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
