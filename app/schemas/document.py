"""Document API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    """A document in the library."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    subject: str
    num_pages: int
    num_chunks: int
    status: str
    error_message: str | None = None
    uploaded_at: datetime


class UploadResponse(BaseModel):
    """Result of an upload."""

    doc_id: str
    filename: str
    status: str
    num_pages: int
    num_chunks: int


class DeleteResponse(BaseModel):
    """Result of a delete."""

    deleted: bool = True
