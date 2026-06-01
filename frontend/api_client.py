"""Thin httpx wrapper over the FastAPI backend."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def health() -> dict:
    r = httpx.get(f"{BACKEND_URL}/health", timeout=10.0)
    r.raise_for_status()
    return r.json()


def ingest(files: list[tuple[str, bytes, str]]) -> dict:
    """files: list of (filename, content_bytes, content_type)."""
    multipart = [("files", (name, content, ctype)) for name, content, ctype in files]
    r = httpx.post(f"{BACKEND_URL}/api/ingest", files=multipart, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def query(question: str, top_k: int | None = None, top_n: int | None = None) -> dict:
    payload = {"question": question, "top_k": top_k, "top_n": top_n}
    r = httpx.post(f"{BACKEND_URL}/api/query", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def query_stream(
    question: str, top_k: int | None = None, top_n: int | None = None
) -> Iterator[dict]:
    """Yield decoded SSE events: {'type': 'token'|'sources'|'done'|'error', ...}."""
    payload = {"question": question, "top_k": top_k, "top_n": top_n}
    with httpx.stream(
        "POST", f"{BACKEND_URL}/api/query/stream", json=payload, timeout=_TIMEOUT
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                yield json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue


def list_documents() -> list[dict]:
    r = httpx.get(f"{BACKEND_URL}/api/documents", timeout=30.0)
    r.raise_for_status()
    return r.json()


def delete_document(doc_id: str) -> dict:
    r = httpx.delete(f"{BACKEND_URL}/api/documents/{doc_id}", timeout=60.0)
    r.raise_for_status()
    return r.json()
