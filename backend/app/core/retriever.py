"""Hybrid retrieval: (BM25 + dense) RRF fusion, then FlashRank reranking.

Pipeline order is hybrid -> rerank: the EnsembleRetriever widens the candidate
pool (Reciprocal Rank Fusion across BM25 + Chroma), then the
ContextualCompressionRetriever narrows it with the cross-encoder reranker.
"""
from __future__ import annotations

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_core.retrievers import BaseRetriever

from ..config import settings
from .bm25 import bm25_index
from .reranker import get_reranker
from .vectorstore import get_vectorstore


def build_base_retriever(top_k: int) -> BaseRetriever:
    """Dense (Chroma) alone, or fused with BM25 when a lexical index exists."""
    dense = get_vectorstore().as_retriever(search_kwargs={"k": top_k})

    bm25 = bm25_index.retriever
    if bm25 is None:
        return dense
    bm25.k = top_k
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[settings.bm25_weight, settings.dense_weight],
    )


def build_retriever(top_k: int | None = None, top_n: int | None = None) -> BaseRetriever:
    """Full hybrid + rerank retriever. Per-request top_k / top_n override config."""
    top_k = top_k or settings.retrieve_k
    base = build_base_retriever(top_k)
    compressor = get_reranker(top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )
