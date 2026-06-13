"""
Query endpoints.

POST /query         — blocking JSON response
POST /query/stream  — Server-Sent Events streaming response

SSE event format:
  data: {"type": "token",    "content": "..."}
  data: {"type": "citation", "citations": [...]}
  data: {"type": "done",     "route": "specific", "from_cache": false}
  data: {"type": "error",    "detail": "..."}
"""

import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import (
    get_qdrant, get_bm25, get_cache_manager,
    get_query_optimizer, get_generator, verify_api_key,
)
from api.schemas import QueryRequest, QueryResponse, CitationModel
from utils.query_classifier import is_overview_query
from utils.embedder import embed_query
from utils.logger import get_logger
from config import GENERATOR

logger = get_logger("api.query")
router = APIRouter()


# ── Shared pipeline logic ─────────────────────────────────────

def _run_specific_pipeline(request: QueryRequest, raw_vec: list[float]):
    """Run stages 7-8 and return (ranked_chunks, rewritten_query)."""
    client    = get_qdrant()
    bm25      = get_bm25()
    optimizer = get_query_optimizer()

    if request.no_optimize:
        from vector_store.retriever import hybrid_search
        from query_optimizer.metadata import QueryFilters
        fused = hybrid_search(
            client=client, bm25_store=bm25,
            query_vector=raw_vec, query_text=request.question,
            top_k=request.top_k,
            **QueryFilters(channel=request.channel, video_id=request.video_id).to_dict(),
        )
        rewritten = request.question
    else:
        optimized = optimizer.run(
            request.question,
            explicit_channel=request.channel,
            explicit_video_id=request.video_id,
        )
        from query_optimizer.retriever import multi_query_retrieve
        fused = multi_query_retrieve(
            client=client, bm25_store=bm25,
            optimized=optimized, top_k=request.top_k,
        )
        rewritten = optimized.rewritten

    if request.no_rerank:
        from reranker.model import _fallback
        ranked = _fallback(fused, request.top_n)
    else:
        from reranker.model import rerank
        ranked = rerank(query=rewritten, candidates=fused, top_n=request.top_n)

    return ranked, rewritten


def _run_overview_pipeline(request: QueryRequest):
    """Build transcript context for overview queries."""
    from generator.transcript_context import build_transcript_context
    from generator.overview_prompt import build_overview_prompts
    bm25 = get_bm25()
    ctx = build_transcript_context(
        bm25,
        video_id=request.video_id,
        channel=request.channel,
    )
    if ctx is None:
        raise HTTPException(status_code=404, detail="No indexed videos found.")
    system_prompt, user_message = build_overview_prompts(request.question, ctx)
    return system_prompt, user_message, ctx


# ── Blocking endpoint ─────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_blocking(
    request: QueryRequest,
    _: None = Depends(verify_api_key),
) -> QueryResponse:
    """Full RAG pipeline — returns complete JSON answer."""
    cache   = get_cache_manager()
    raw_vec = embed_query(request.question)
    filters = {k: v for k, v in {"channel": request.channel, "video_id": request.video_id}.items() if v}

    # Cache lookup
    if not request.no_cache:
        cached = cache.get_query(request.question, raw_vec, **filters)
        if cached is not None:
            return QueryResponse(
                question=request.question,
                answer=cached.get("answer", ""),
                citations=[CitationModel(**c) for c in cached.get("citations", [])],
                is_confident=cached.get("is_confident", True),
                from_cache=True,
                route=cached.get("route", "specific"),
                model=GENERATOR.model,
            )

    # Detect route
    is_overview = (
        request.force_overview
        or (not request.force_specific and is_overview_query(request.question, raw_vec))
    )

    if is_overview:
        # Overview route
        from generator.llm import generate
        system_prompt, user_message, ctx = _run_overview_pipeline(request)
        answer = generate(system_prompt, user_message, stream=False)
        citations = []
        route = "overview"
        is_confident = True
    else:
        # Specific RAG route
        ranked, rewritten = _run_specific_pipeline(request, raw_vec)
        if not ranked:
            raise HTTPException(status_code=404, detail="No relevant content found.")
        generator = get_generator()
        result = generator.run(rewritten, ranked)
        answer = result.answer
        citations = result.citations
        is_confident = result.is_confident
        route = "specific"

    # Build response
    citation_models = [
        CitationModel(
            index=c.index, video_title=c.video_title, channel=c.channel,
            timestamp_label=c.timestamp_label, url_with_timestamp=c.url_with_timestamp,
        )
        for c in citations
    ] if citations else []

    # Cache result
    if not request.no_cache:
        cache.set_query(request.question, raw_vec, {
            "answer": answer,
            "citations": [c.model_dump() for c in citation_models],
            "is_confident": is_confident,
            "route": route,
        }, **filters)

    return QueryResponse(
        question=request.question,
        answer=answer,
        citations=citation_models,
        is_confident=is_confident,
        from_cache=False,
        route=route,
        model=GENERATOR.model,
    )


# ── Streaming SSE endpoint ────────────────────────────────────

@router.post("/query/stream", tags=["Query"])
async def query_stream(
    request: QueryRequest,
    _: None = Depends(verify_api_key),
):
    """
    Stream answer tokens via Server-Sent Events.

    Event types:
      {"type": "token",    "content": "..."}   — answer chunk
      {"type": "citation", "citations": [...]}  — source citations
      {"type": "done",     "route": "...",  "from_cache": bool}
      {"type": "error",    "detail": "..."}
    """

    async def event_gen():
        try:
            cache   = get_cache_manager()
            raw_vec = embed_query(request.question)
            filters = {k: v for k, v in {"channel": request.channel, "video_id": request.video_id}.items() if v}

            # Cache hit — replay cached answer as tokens
            if not request.no_cache:
                cached = cache.get_query(request.question, raw_vec, **filters)
                if cached is not None:
                    for word in cached.get("answer", "").split(" "):
                        yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                        await asyncio.sleep(0)
                    yield f"data: {json.dumps({'type': 'citation', 'citations': cached.get('citations', [])})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'route': cached.get('route', 'specific'), 'from_cache': True})}\n\n"
                    return

            # Detect route
            is_overview = (
                request.force_overview
                or (not request.force_specific and is_overview_query(request.question, raw_vec))
            )

            if is_overview:
                from generator.llm import generate
                system_prompt, user_message, ctx = _run_overview_pipeline(request)
                full_answer = ""
                for token in generate(system_prompt, user_message, stream=True):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'type': 'citation', 'citations': []})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'route': 'overview', 'from_cache': False})}\n\n"
                route = "overview"
                citations_data = []
                is_confident = True

            else:
                ranked, rewritten = _run_specific_pipeline(request, raw_vec)
                if not ranked:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'No relevant content found.'})}\n\n"
                    return

                generator = get_generator()
                token_gen, result_stub = generator.stream(rewritten, ranked)

                full_answer = ""
                for token in token_gen:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)

                result_stub.answer = full_answer
                citations_data = [
                    {
                        "index": c.index, "video_title": c.video_title,
                        "channel": c.channel, "timestamp_label": c.timestamp_label,
                        "url_with_timestamp": c.url_with_timestamp,
                    }
                    for c in result_stub.citations
                ]
                yield f"data: {json.dumps({'type': 'citation', 'citations': citations_data})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'route': 'specific', 'from_cache': False})}\n\n"
                route = "specific"
                is_confident = result_stub.is_confident

            # Cache
            if not request.no_cache:
                cache.set_query(request.question, raw_vec, {
                    "answer": full_answer,
                    "citations": citations_data if not is_overview else [],
                    "is_confident": is_confident,
                    "route": route,
                }, **filters)

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )