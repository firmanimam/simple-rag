# RAG_simple — PDF/Image RAG with Hybrid Retrieval + Reranking

A Retrieval-Augmented Generation system that ingests **PDFs and images**, turns
everything into text (PyMuPDF text extraction, GPT-4o vision captions, Tesseract
OCR), indexes it in **ChromaDB**, retrieves with **hybrid search** (BM25 + dense
fused via Reciprocal Rank Fusion), reranks with a **FlashRank** cross-encoder,
and answers questions through a **Streamlit** chat UI backed by a **FastAPI**
service.

```
query → [BM25 top-k] + [Chroma dense top-k]
      → EnsembleRetriever (RRF fuse)
      → FlashRank rerank → top-n
      → grounded prompt → gpt-4o → answer + citations
```

## Stack

| Layer            | Choice                                             |
| ---------------- | -------------------------------------------------- |
| Backend API      | FastAPI + Uvicorn                                  |
| Frontend         | Streamlit                                          |
| Orchestration    | LangChain (1.x) + `langchain-classic`              |
| LLM / Vision     | OpenAI `gpt-4o` / `gpt-4o-mini`                    |
| Embeddings       | OpenAI `text-embedding-3-small`                    |
| Vector DB        | ChromaDB (persistent)                              |
| Sparse retrieval | BM25 (`rank-bm25`, in-memory, rebuilt from Chroma) |
| Hybrid fusion    | LangChain `EnsembleRetriever` (RRF)                |
| Reranker         | FlashRank `ms-marco-MiniLM-L-12-v2` (local, free)  |
| PDF parsing      | PyMuPDF                                            |
| OCR / vision     | Tesseract + GPT-4o vision                          |

## Prerequisites

System binaries (macOS / Homebrew):

```bash
brew install tesseract poppler
```

- **Tesseract** powers OCR for scanned pages / images.
- **Poppler** backs `pdf2image` (optional fallback rendering).

Python deps are managed by **uv** (already pinned in `pyproject.toml` / `uv.lock`).

## Setup

```bash
# 1. Install Python deps
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

> The app reads `.env` automatically (pydantic-settings). All keys are optional
> except `OPENAI_API_KEY`, which is required for embeddings, answers, and vision
> captions.

## Run

Two terminals:

```bash
# Terminal 1 — backend (http://localhost:8000, Swagger at /docs)
uv run uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:8501)
uv run streamlit run frontend/streamlit_app.py
```

Then open the Streamlit app, upload PDFs/images in the sidebar, click **Ingest**,
and start asking questions. If the frontend runs on a different host, point it at
the backend with `BACKEND_URL=http://host:8000`.

## API

| Method   | Path                  | Purpose                                |
| -------- | --------------------- | -------------------------------------- |
| `GET`    | `/health`             | Liveness + indexed chunk count         |
| `POST`   | `/api/ingest`         | Upload & index files (multipart)       |
| `POST`   | `/api/query`          | Ask a question → `{answer, sources[]}` |
| `POST`   | `/api/query/stream`   | SSE token stream + final sources       |
| `GET`    | `/api/documents`      | List indexed documents                 |
| `DELETE` | `/api/documents/{id}` | Remove a document & its vectors        |

`sources[]` entries: `{filename, page, type, snippet, score, doc_id}` where
`type` ∈ `text` / `ocr` / `caption`.

## How ingestion handles images & scanned PDFs

Everything becomes text before indexing, so retrieval + reranking stay uniform:

- **Digital PDF page** → PyMuPDF text layer (`type=text`).
- **Scanned page** (no/empty text layer) → rendered to an image, then OCR +
  vision caption (`type=ocr`).
- **Embedded / standalone images** → GPT-4o vision caption + Tesseract OCR
  (`type=caption`).

Each chunk keeps `source`, `page`, and `type` metadata for precise citations.

## Hybrid search lifecycle

Chroma is the single source of truth for chunk text. `BM25Retriever` is
in-memory and **not** persisted — it is rebuilt from the Chroma corpus on
startup (FastAPI lifespan) and refreshed after every ingest/delete so the
lexical and dense indexes stay in sync.

## Configuration

All tunables live in `.env` (see `.env.example`): models, `RETRIEVE_K`,
`RERANK_N`, BM25/dense weights, chunk size/overlap, and vision/OCR toggles.

## Project layout

```
backend/app/
  main.py            FastAPI app + CORS + lifespan (BM25 build)
  config.py          pydantic-settings
  api/routes/        ingest, query, documents
  core/              loaders, vision, chunking, vectorstore,
                     bm25, retriever, reranker, rag_chain
  models/schemas.py  Pydantic request/response models
frontend/
  streamlit_app.py   chat UI, upload, sources, doc management
  api_client.py      httpx wrapper over the backend
data/                uploads/ + chroma_db/ (gitignored)
tests/
```

## Tests

```bash
uv run pytest -q
```
