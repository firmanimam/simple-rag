"""Offline tests: PDF text extraction, chunking, and BM25 build.

These do not require an OpenAI key — they exercise the parsing/retrieval plumbing
that runs before embeddings.
"""
from __future__ import annotations

import fitz
import pytest
from langchain_community.retrievers import BM25Retriever

from backend.app.core.chunking import chunk_documents
from backend.app.core.loaders import load_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Invoice number INV-12345 issued to Acme Corp.")
    page.insert_text((72, 100), "The total amount due is 4200 dollars, net 30 days.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Refund policy: returns accepted within 14 days of delivery.")
    doc.save(path)
    doc.close()
    return path


def test_load_pdf_extracts_text(sample_pdf):
    docs = load_pdf(sample_pdf, source="sample.pdf", doc_id="doc-1")
    assert len(docs) == 2  # two pages with a real text layer
    assert all(d.metadata["type"] == "text" for d in docs)
    assert all(d.metadata["doc_id"] == "doc-1" for d in docs)
    assert {d.metadata["page"] for d in docs} == {1, 2}
    assert "INV-12345" in docs[0].page_content


def test_chunking_preserves_metadata(sample_pdf):
    docs = load_pdf(sample_pdf, source="sample.pdf", doc_id="doc-1")
    chunks = chunk_documents(docs)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.metadata["doc_id"] == "doc-1"
        assert c.metadata["source"] == "sample.pdf"
        assert "chunk_index" in c.metadata


def test_bm25_retrieves_by_keyword(sample_pdf):
    docs = load_pdf(sample_pdf, source="sample.pdf", doc_id="doc-1")
    chunks = chunk_documents(docs)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 2
    hits = bm25.invoke("refund policy returns")
    assert hits
    assert any("Refund policy" in h.page_content for h in hits)
