"""Persistent Chroma vector store: the single source of truth for chunk text.

Chroma persists embeddings + chunk text + metadata to local disk. The BM25
index (see bm25.py) is rebuilt from this corpus, so we never keep a second
docstore.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..config import settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    # A placeholder key lets the Chroma handle (and thus list/delete/count and the
    # BM25 rebuild, which never embed) construct without OPENAI_API_KEY set.
    # Real embedding calls (ingest/query) still require a valid key and will 401.
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or "sk-not-configured",
    )


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_path),
    )


def add_documents(chunks: list[Document]) -> list[str]:
    """Embed + persist chunks. Stamps created_at on any chunk missing it."""
    if not chunks:
        return []
    now = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []
    for chunk in chunks:
        chunk.metadata.setdefault("created_at", now)
        ids.append(str(uuid.uuid4()))
    vs = get_vectorstore()
    vs.add_documents(chunks, ids=ids)
    return ids


def delete_document(doc_id: str) -> int:
    """Delete every chunk belonging to a doc_id. Returns chunks removed."""
    vs = get_vectorstore()
    existing = vs.get(where={"doc_id": doc_id})
    ids = existing.get("ids", [])
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def get_corpus_documents() -> list[Document]:
    """Reconstruct all chunks as LangChain Documents (used to build BM25)."""
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs: list[Document] = []
    for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    return docs


def count_chunks() -> int:
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return len(get_corpus_documents())


def list_documents() -> list[dict]:
    """Aggregate chunks into one record per source document."""
    vs = get_vectorstore()
    data = vs.get(include=["metadatas"])
    agg: dict[str, dict] = {}
    for meta in data.get("metadatas", []):
        meta = meta or {}
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        rec = agg.setdefault(
            doc_id,
            {
                "id": doc_id,
                "filename": meta.get("source", "unknown"),
                "pages": 0,
                "chunks": 0,
                "created_at": meta.get("created_at"),
            },
        )
        rec["chunks"] += 1
        page = meta.get("page")
        if isinstance(page, int):
            rec["pages"] = max(rec["pages"], page)
        if not rec["created_at"] and meta.get("created_at"):
            rec["created_at"] = meta.get("created_at")
    return sorted(agg.values(), key=lambda r: r.get("created_at") or "")
