"""
Health, stats, and video listing endpoints.

GET /health    — liveness/readiness check
GET /stats     — per-layer statistics
GET /videos    — list all indexed videos
"""

from fastapi import APIRouter, Depends
from api.deps import get_qdrant, get_bm25, get_cache_manager, verify_api_key
from api.schemas import HealthResponse, VideoItem

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    # Qdrant
    qdrant_ok = False
    try:
        get_qdrant().get_collections()
        qdrant_ok = True
    except Exception:
        pass

    # Redis
    from cache.redis_client import is_available
    redis_ok = is_available()

    # LLM API key present
    from config import GENERATOR
    llm_ok = bool(GENERATOR.groq_api_key)

    all_critical_ok = qdrant_ok
    status = "healthy" if all_critical_ok else "unhealthy"
    if all_critical_ok and not (redis_ok and llm_ok):
        status = "degraded"

    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        redis=redis_ok,
        llm_api=llm_ok,
    )


@router.get("/stats", tags=["Health"])
async def stats(_: None = Depends(verify_api_key)):
    from vector_store.client import get_collection_stats
    from config import RERANKER, GENERATOR, EMBEDDING

    qdrant_stats = {}
    try:
        qdrant_stats = get_collection_stats(get_qdrant())
    except Exception as e:
        qdrant_stats = {"error": str(e)}

    return {
        "qdrant":    qdrant_stats,
        "bm25":      get_bm25().stats(),
        "cache":     get_cache_manager().stats(),
        "embedding": {"backend": EMBEDDING.backend, "model": EMBEDDING.model_name},
        "reranker":  {"backend": RERANKER.backend, "top_n": RERANKER.top_n, "enabled": RERANKER.enabled},
        "generator": {"model": GENERATOR.model, "max_tokens": GENERATOR.max_tokens},
    }


@router.get("/videos", tags=["Videos"])
async def list_videos(_: None = Depends(verify_api_key)):
    bm25 = get_bm25()
    videos = []
    for video_id, chunk_ids in bm25._video_chunks.items():
        if chunk_ids and chunk_ids[0] in bm25._payload_map:
            p = bm25._payload_map[chunk_ids[0]]
            videos.append(VideoItem(
                video_id=p.video_id,
                video_title=p.video_title,
                channel=p.channel,
                video_url=p.video_url,
                chunk_count=len(chunk_ids),
            ))
    return {"total": len(videos), "videos": videos}


@router.delete("/videos/{video_id}", tags=["Videos"])
async def delete_video(video_id: str, _: None = Depends(verify_api_key)):
    from vector_store.indexer import delete_video as _delete
    _delete(get_qdrant(), get_bm25(), video_id)
    return {"message": f"Deleted video '{video_id}' from all indexes."}