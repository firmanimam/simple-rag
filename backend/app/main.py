"""FastAPI application: CORS, routers, health, and BM25 startup build."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import documents, ingest, query
from .core.bm25 import bm25_index
from .core.vectorstore import count_chunks
from .models.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the in-memory BM25 index from whatever Chroma already persisted.
    try:
        n = bm25_index.refresh()
        logger.info("Startup: BM25 index built from %d chunks.", n)
    except Exception:
        logger.exception("Startup: failed to build BM25 index (continuing).")
    yield


app = FastAPI(
    title="RAG Pipeline API",
    version="0.1.0",
    description="PDF/image ingestion + hybrid retrieval + reranking RAG.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(documents.router, prefix="/api", tags=["documents"])


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        count = count_chunks()
    except Exception:
        count = 0
    return HealthResponse(status="ok", doc_count=count)
