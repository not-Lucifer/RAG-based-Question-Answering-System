"""Document library endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import document_service
from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError
from app.schemas.document import DeleteResponse, DocumentOut, UploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])
_READ_CHUNK = 1024 * 1024


def _read_limited(upload: UploadFile) -> bytes:
    """Read the upload, failing fast once it exceeds MAX_UPLOAD_MB."""
    limit = get_settings().max_upload_mb * 1024 * 1024
    buf = bytearray()
    while chunk := upload.file.read(_READ_CHUNK):
        buf.extend(chunk)
        if len(buf) > limit:
            raise FileTooLargeError(f"File exceeds the {get_settings().max_upload_mb} MB limit.")
    return bytes(buf)


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(..., description="A text-based PDF"),
    subject: str | None = Form(None, max_length=100),
    svc: DocumentService = Depends(document_service),
) -> UploadResponse:
    """Upload and index a PDF."""
    data = _read_limited(file)
    return svc.upload(file.filename or "document.pdf", data, subject, file.content_type)


@router.get("", response_model=list[DocumentOut])
def list_documents(svc: DocumentService = Depends(document_service)) -> list[DocumentOut]:
    """List all documents."""
    return list(svc.list())


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, svc: DocumentService = Depends(document_service)) -> DeleteResponse:
    """Delete a document's file, vectors and row."""
    svc.delete(doc_id)
    return DeleteResponse()


@router.post("/{doc_id}/reindex", response_model=DocumentOut)
def reindex_document(doc_id: str, svc: DocumentService = Depends(document_service)) -> DocumentOut:
    """Re-run ingestion for a document."""
    return svc.reindex(doc_id)
