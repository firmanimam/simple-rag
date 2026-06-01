"""In-memory BM25 lexical index, rebuilt from the Chroma corpus.

BM25Retriever does not persist to disk: it builds from a Python list of
Documents. Chroma stays the source of truth, so we (re)build BM25 on startup
and refresh it after every ingest / delete to keep the two in sync.
"""
from __future__ import annotations

import logging
import threading

from langchain_community.retrievers import BM25Retriever

from ..config import settings
from .vectorstore import get_corpus_documents

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self) -> None:
        self._retriever: BM25Retriever | None = None
        self._lock = threading.Lock()

    def refresh(self) -> int:
        """Rebuild the BM25 index from the current Chroma corpus.

        Returns the number of documents indexed.
        """
        docs = get_corpus_documents()
        with self._lock:
            if not docs:
                self._retriever = None
                logger.info("BM25 index empty (no documents in corpus).")
                return 0
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = settings.retrieve_k
            self._retriever = retriever
        logger.info("BM25 index rebuilt with %d documents.", len(docs))
        return len(docs)

    @property
    def retriever(self) -> BM25Retriever | None:
        return self._retriever


# Module-level singleton shared across the app.
bm25_index = BM25Index()
