"""Split loaded Documents into overlapping chunks, preserving metadata."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents; each chunk inherits its parent's metadata.

    A `chunk_index` is added per source document so chunks remain orderable.
    """
    splitter = _splitter()
    chunks = splitter.split_documents(docs)
    # Number chunks within each (doc_id, page) so citations stay stable.
    counters: dict[tuple, int] = {}
    for chunk in chunks:
        key = (chunk.metadata.get("doc_id"), chunk.metadata.get("page"))
        idx = counters.get(key, 0)
        chunk.metadata["chunk_index"] = idx
        counters[key] = idx + 1
    return chunks
