"""GET /api/documents and DELETE /api/documents/{id} — manage indexed docs."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ...core.bm25 import bm25_index
from ...core.vectorstore import delete_document, list_documents
from ...models.schemas import DeleteResponse, DocumentInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=list[DocumentInfo])
async def get_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**rec) for rec in list_documents()]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str) -> DeleteResponse:
    removed = delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id}")
    # Rebuild the lexical index now that Chroma changed.
    bm25_index.refresh()
    return DeleteResponse(deleted=True, doc_id=doc_id, chunks_removed=removed)
