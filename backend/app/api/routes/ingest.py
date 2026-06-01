"""POST /api/ingest — upload files, extract text, chunk, embed, index."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...core.bm25 import bm25_index
from ...core.chunking import chunk_documents
from ...core.loaders import SUPPORTED_EXTS, load_file
from ...core.vectorstore import add_documents
from ...config import settings
from ...models.schemas import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    doc_ids: list[str] = []
    filenames: list[str] = []
    total_chunks = 0

    for upload in files:
        filename = upload.filename or "upload"
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for {filename}. "
                f"Supported: {sorted(SUPPORTED_EXTS)}",
            )

        doc_id = str(uuid.uuid4())
        dest = settings.upload_path / f"{doc_id}{ext}"
        contents = await upload.read()
        dest.write_bytes(contents)

        try:
            docs = load_file(dest, source=filename, doc_id=doc_id)
        except Exception as exc:
            logger.exception("Failed to load %s", filename)
            raise HTTPException(status_code=422, detail=f"Failed to parse {filename}: {exc}")

        if not docs:
            logger.warning("No extractable content in %s", filename)
            continue

        chunks = chunk_documents(docs)
        add_documents(chunks)

        doc_ids.append(doc_id)
        filenames.append(filename)
        total_chunks += len(chunks)

    if not doc_ids:
        raise HTTPException(status_code=422, detail="No extractable content found in uploads.")

    # Keep the lexical index in sync with Chroma.
    bm25_index.refresh()

    return IngestResponse(doc_ids=doc_ids, chunks_indexed=total_chunks, filenames=filenames)
