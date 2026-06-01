"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    doc_count: int


class IngestResponse(BaseModel):
    doc_ids: list[str]
    chunks_indexed: int
    filenames: list[str]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, description="Candidates per retriever before fusion")
    top_n: int | None = Field(default=None, description="Final chunks after reranking")


class Source(BaseModel):
    filename: str
    page: int | None = None
    type: str = "text"          # text | ocr | caption
    snippet: str
    score: float | None = None
    doc_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: int | None = None
    chunks: int
    created_at: str | None = None


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str
    chunks_removed: int
