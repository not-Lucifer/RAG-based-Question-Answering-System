"""Custom exceptions and (lazily imported) FastAPI exception handlers.

The RAG core imports the exception classes from here, so this module must not
import FastAPI at import time; handlers are registered via
:func:`register_exception_handlers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI


class AppError(Exception):
    """Base class for all expected, user-facing errors."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500
    default_detail: str = "Unexpected error."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class InvalidFileError(AppError):
    code, status_code, default_detail = "INVALID_FILE", 400, "The uploaded file is not a valid PDF."


class FileTooLargeError(AppError):
    code, status_code, default_detail = "FILE_TOO_LARGE", 413, "The uploaded file is too large."


class EmptyPDFError(AppError):
    code, status_code = "EMPTY_PDF", 422
    default_detail = "No selectable text found in the PDF (it may be scanned). OCR is not supported."


class DuplicateDocumentError(AppError):
    code, status_code, default_detail = "DUPLICATE_DOCUMENT", 409, "This document is already indexed."


class NotFoundError(AppError):
    code, status_code, default_detail = "NOT_FOUND", 404, "Resource not found."


class LLMUnavailableError(AppError):
    code, status_code, default_detail = "LLM_UNAVAILABLE", 503, "The language model is unavailable."


class IndexMismatchError(AppError):
    code, status_code = "INDEX_MISMATCH", 409
    default_detail = (
        "The vector index was built with a different embedding model. "
        "Run `python scripts/reset_index.py --reindex` to rebuild it."
    )


class EmbeddingError(AppError):
    code, status_code, default_detail = "EMBEDDING_UNAVAILABLE", 503, "The embedding model is unavailable."


class VectorStoreError(AppError):
    code, status_code, default_detail = "VECTORSTORE_ERROR", 500, "Vector store operation failed."


class IngestionError(AppError):
    code, status_code, default_detail = "INGESTION_FAILED", 500, "Document ingestion failed."


def register_exception_handlers(app: FastAPI) -> None:
    """Map exceptions to ``{"error": code, "detail": message}`` JSON responses."""
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    from app.core.logging import get_logger

    log = get_logger("app.errors")

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        log.info("request failed", extra={"fields": {"code": exc.code}})
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg')}" for err in exc.errors()
        )
        return JSONResponse(status_code=422, content={"error": "VALIDATION_ERROR", "detail": detail})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Log the full error server-side, but never leak internals to clients.
        log.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500, content={"error": "INTERNAL_ERROR", "detail": "Unexpected server error."}
        )
