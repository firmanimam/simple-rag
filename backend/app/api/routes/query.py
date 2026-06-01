"""POST /api/query and /api/query/stream — answer questions over the corpus."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...core.rag_chain import answer, stream_answer
from ...models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        result = answer(req.question, top_k=req.top_k, top_n=req.top_n)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@router.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Server-Sent Events: token stream followed by a final sources event."""

    def event_stream():
        try:
            sources, tokens = stream_answer(req.question, top_k=req.top_k, top_n=req.top_n)
            for tok in tokens:
                yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
            payload = {"type": "sources", "sources": [s.model_dump() for s in sources]}
            yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:  # surface errors to the client over SSE
            logger.exception("Streaming query failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
