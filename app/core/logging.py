"""Structured (key=value) logging with secret redaction."""

from __future__ import annotations

import logging
import re
import sys

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***"),  # OpenAI-style keys
    (re.compile(r"(?i)\b(bearer)\s+[^\s,'\"]+"), r"\1 ***"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,'\"]+"), r"\1\2***"),
]
_CONFIGURED = False


def redact(text: str) -> str:
    """Mask anything that looks like an API key, bearer token or password."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Formats records as ``time level logger msg key=value ...`` and redacts secrets."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = getattr(record, "fields", None)
        if isinstance(extras, dict) and extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return redact(base)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once; later calls only adjust the level."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
    root.handlers = [handler]
    for noisy in ("httpx", "httpcore", "chromadb.telemetry", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Pass structured fields via ``extra={"fields": {...}}``."""
    return logging.getLogger(name)
