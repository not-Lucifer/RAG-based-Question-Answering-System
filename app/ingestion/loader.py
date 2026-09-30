"""PDF loading with PyMuPDF: one LangChain Document per page."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from app.core.exceptions import EmptyPDFError, InvalidFileError
from app.core.logging import get_logger

log = get_logger(__name__)


def load_pdf(path: Path, source_name: str | None = None) -> list[Document]:
    """Extract text page by page.

    Args:
        path: Location of the PDF on disk.
        source_name: Display filename stored as ``source`` metadata. Defaults to
            ``path.name`` (uploads are stored as ``<doc_id>.pdf``, so the service
            passes the original filename here).

    Returns:
        One Document per page with metadata ``{source, page}`` (1-based page).

    Raises:
        InvalidFileError: The file cannot be opened as a PDF.
        EmptyPDFError: No page contains extractable text (likely scanned).
    """
    path = Path(path)
    source = source_name or path.name
    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted and not pdf.authenticate(""):
                raise InvalidFileError("The PDF is password-protected.")
            pages = [
                Document(page_content=page.get_text("text") or "", metadata={"source": source, "page": i + 1})
                for i, page in enumerate(pdf)
            ]
    except InvalidFileError:
        raise
    except Exception as exc:  # PyMuPDF raises several error types for corrupt files
        log.warning("failed to open pdf", extra={"fields": {"source": source, "error": type(exc).__name__}})
        raise InvalidFileError("The file could not be read as a PDF.") from exc

    if not any(doc.page_content.strip() for doc in pages):
        raise EmptyPDFError()
    log.info("loaded pdf", extra={"fields": {"source": source, "pages": len(pages)}})
    return pages
