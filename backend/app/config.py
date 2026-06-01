"""Typed application settings loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    vision_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # --- Storage paths ---
    chroma_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    collection_name: str = "rag_documents"

    # --- Retrieval / reranking ---
    retrieve_k: int = 20        # candidates pulled per retriever before fusion
    rerank_n: int = 5           # final chunks after reranking
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Vision / OCR ---
    enable_vision: bool = True   # use GPT-4o vision captions for images
    enable_ocr: bool = True      # use Tesseract OCR
    # A page is treated as "scanned" (needs OCR/vision) if its extracted
    # text layer is shorter than this many characters.
    scanned_text_threshold: int = 50

    # --- Optional Cohere reranker ---
    cohere_api_key: str = ""

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
