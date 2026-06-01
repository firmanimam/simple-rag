"""Grounded RAG chain: retrieve -> build context -> LLM answer + citations."""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings
from ..models.schemas import Source
from .retriever import build_retriever

SYSTEM_PROMPT = """You are a precise assistant answering questions about the user's documents.

Rules:
- Answer ONLY using the provided context below. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Cite the sources you used inline with bracketed numbers like [1], [2] that match the context blocks.
- Be concise and factual."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


@lru_cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key or None,
        streaming=True,
    )


def retrieve(question: str, top_k: int | None = None, top_n: int | None = None) -> list[Document]:
    retriever = build_retriever(top_k=top_k, top_n=top_n)
    return retriever.invoke(question)


def format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        loc = f"{src}" + (f", p.{page}" if page else "")
        blocks.append(f"[{i}] (source: {loc})\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "(no documents retrieved)"


def build_sources(docs: list[Document]) -> list[Source]:
    sources: list[Source] = []
    for doc in docs:
        meta = doc.metadata or {}
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        score = meta.get("relevance_score")
        sources.append(
            Source(
                filename=meta.get("source", "unknown"),
                page=meta.get("page"),
                type=meta.get("type", "text"),
                snippet=snippet,
                score=float(score) if score is not None else None,
                doc_id=meta.get("doc_id"),
            )
        )
    return sources


def answer(question: str, top_k: int | None = None, top_n: int | None = None) -> dict:
    """Non-streaming answer. Returns {'answer': str, 'sources': list[Source]}."""
    docs = retrieve(question, top_k=top_k, top_n=top_n)
    messages = PROMPT.format_messages(context=format_context(docs), question=question)
    result = get_llm().invoke(messages)
    return {"answer": result.content, "sources": build_sources(docs)}


def stream_answer(
    question: str, top_k: int | None = None, top_n: int | None = None
) -> tuple[list[Source], Iterator[str]]:
    """Retrieve first (so sources are known up front), then stream answer tokens.

    Returns (sources, token_iterator).
    """
    docs = retrieve(question, top_k=top_k, top_n=top_n)
    sources = build_sources(docs)
    messages = PROMPT.format_messages(context=format_context(docs), question=question)

    def token_iter() -> Iterator[str]:
        for chunk in get_llm().stream(messages):
            text = chunk.content
            if text:
                yield text

    return sources, token_iter()
