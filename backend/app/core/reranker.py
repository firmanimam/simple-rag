"""Cross-encoder reranker (FlashRank). Swappable for Cohere / bge.

FlashRank is local and free; the model downloads once on first use.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_community.document_compressors import FlashrankRerank

from ..config import settings


@lru_cache
def _ranker():
    # Imported lazily so importing this module never triggers a model download.
    from flashrank import Ranker

    return Ranker(model_name=settings.reranker_model)


def get_reranker(top_n: int | None = None) -> FlashrankRerank:
    """Build a FlashRank compressor reusing one cached underlying Ranker."""
    return FlashrankRerank(
        client=_ranker(),
        model=settings.reranker_model,
        top_n=top_n or settings.rerank_n,
    )
