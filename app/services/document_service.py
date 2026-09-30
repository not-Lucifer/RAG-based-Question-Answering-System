"""Document library operations: upload (validate, dedupe, ingest), list, delete, reindex."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    AppError,
    DuplicateDocumentError,
    FileTooLargeError,
    IngestionError,
    InvalidFileError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.db import crud
from app.db.models import DocumentRecord, DocumentStatus
from app.ingestion.chunker import DEFAULT_SUBJECT
from app.ingestion.pipeline import ingest_file
from app.schemas.document import DocumentOut, UploadResponse
from app.vectorstore.store import delete_doc

log = get_logger(__name__)
PDF_MAGIC = b"%PDF-"
_UNSAFE_CHARS = re.compile(r"[^\w\s.\-()\[\]&+,]")


def sanitize_filename(filename: str | None) -> str:
    """Display-safe basename: strips directories, control/unsafe characters, limits length."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = _UNSAFE_CHARS.sub("_", name).strip(" .") or "document.pdf"
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "pdf"
    return f"{stem[:150]}.{ext.lower()[:10]}"


def sanitize_subject(subject: str | None) -> str:
    """Trimmed subject tag (defaults to 'General')."""
    subject = re.sub(r"\s+", " ", (subject or "")).strip()[:100]
    return subject or DEFAULT_SUBJECT


def validate_pdf(filename: str, data: bytes, content_type: str | None = None) -> None:
    """Reject non-PDF or oversized uploads.

    Raises:
        InvalidFileError: wrong extension, MIME type or magic bytes, or empty file.
        FileTooLargeError: larger than ``MAX_UPLOAD_MB``.
    """
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the {get_settings().max_upload_mb} MB limit.")
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileError("Only .pdf files are accepted.")
    allowed_types = {None, "", "application/pdf", "application/x-pdf", "application/octet-stream"}
    if content_type not in allowed_types:
        raise InvalidFileError("Only PDF files are accepted.")
    if not data or not data[:1024].lstrip().startswith(PDF_MAGIC):
        raise InvalidFileError("The file is not a valid PDF.")


class DocumentService:
    """Business logic for the document library. Owns DB transactions."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    # ------------------------------------------------------------------ helpers
    def _path(self, doc_id: str) -> Path:
        return self.settings.raw_dir / f"{doc_id}.pdf"

    def _get(self, doc_id: str) -> DocumentRecord:
        record = crud.get_document(self.db, doc_id)
        if record is None:
            raise NotFoundError(f"Document '{doc_id}' not found.")
        return record

    def _run_ingest(self, record: DocumentRecord) -> None:
        """Ingest the stored file and update the row; mark FAILED on any error."""
        try:
            result = ingest_file(
                self._path(record.id), record.id, record.subject, source_name=record.filename
            )
        except AppError as exc:
            self._fail(record, exc.detail)
            raise
        except Exception as exc:
            log.exception("ingestion crashed", extra={"fields": {"doc_id": record.id}})
            self._fail(record, "Unexpected error while processing the document.")
            raise IngestionError() from exc
        crud.mark_document_ready(self.db, record, result.num_pages, result.num_chunks)
        self.db.commit()

    def _fail(self, record: DocumentRecord, message: str) -> None:
        self.db.rollback()
        crud.mark_document_failed(self.db, record, message)
        self.db.commit()
        try:
            delete_doc(record.id)  # no half-indexed documents
        except AppError:
            log.warning("cleanup of failed document chunks failed", extra={"fields": {"doc_id": record.id}})

    def _remove(self, record: DocumentRecord) -> None:
        delete_doc(record.id)
        self._path(record.id).unlink(missing_ok=True)
        self.db.delete(record)
        self.db.commit()

    # ------------------------------------------------------------------ public API
    def upload(
        self, filename: str, data: bytes, subject: str | None = None, content_type: str | None = None
    ) -> UploadResponse:
        """Validate, dedupe (SHA-256), store and ingest an uploaded PDF."""
        name = sanitize_filename(filename)
        validate_pdf(name, data, content_type)
        file_hash = hashlib.sha256(data).hexdigest()

        existing = crud.get_document_by_hash(self.db, file_hash)
        if existing is not None:
            if existing.status != DocumentStatus.FAILED.value:
                raise DuplicateDocumentError(f"'{existing.filename}' is already in your library.")
            self._remove(existing)  # a previous attempt failed: retry from scratch

        doc_id = str(uuid.uuid4())
        path = self._path(doc_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as exc:
            log.exception("saving upload failed")
            raise IngestionError("Could not save the uploaded file.") from exc

        try:
            record = crud.create_document(self.db, doc_id, name, sanitize_subject(subject), file_hash)
            self.db.commit()
        except IntegrityError as exc:  # concurrent upload of the same file
            self.db.rollback()
            path.unlink(missing_ok=True)
            raise DuplicateDocumentError("This document is already in your library.") from exc

        self._run_ingest(record)
        return UploadResponse(
            doc_id=record.id,
            filename=record.filename,
            status=record.status,
            num_pages=record.num_pages,
            num_chunks=record.num_chunks,
        )

    def list(self) -> Sequence[DocumentOut]:
        """All documents, newest first."""
        return [DocumentOut.model_validate(r) for r in crud.list_documents(self.db)]

    def get(self, doc_id: str) -> DocumentOut:
        """One document."""
        return DocumentOut.model_validate(self._get(doc_id))

    def delete(self, doc_id: str) -> None:
        """Remove the document's vectors, stored file and row."""
        self._remove(self._get(doc_id))
        log.info("deleted document", extra={"fields": {"doc_id": doc_id}})

    def reindex(self, doc_id: str) -> DocumentOut:
        """Re-run ingestion from the stored file (e.g. after changing chunking settings)."""
        record = self._get(doc_id)
        if not self._path(doc_id).exists():
            self._fail(record, "The original file is missing; delete and upload it again.")
            raise NotFoundError("The original file for this document is missing.")
        record.status = DocumentStatus.PROCESSING.value
        self.db.commit()
        self._run_ingest(record)
        return DocumentOut.model_validate(record)
