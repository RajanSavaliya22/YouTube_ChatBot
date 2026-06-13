"""
Indexing endpoints.

POST   /index              — index a YouTube video (async background job)
GET    /index/status/{id}  — poll job status
DELETE /index/{video_id}   — remove a video from the index
"""

import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends
from api.deps import get_qdrant, get_bm25, get_cache_manager, verify_api_key
from api.schemas import IndexRequest, IndexResponse
from utils.logger import get_logger

logger = get_logger("api.index")
router = APIRouter()

# In-memory job store (sufficient for single-worker free tier)
_jobs: dict[str, dict] = {}


def _run_pipeline(
    job_id: str,
    url: str,
    force_reindex: bool,
    force_whisper: bool,
) -> None:
    """Full ingestion pipeline run as a background task."""
    _jobs[job_id] = {"status": "running", "step": "transcript", "error": None}
    try:
        from pipeline.stage01_transcript import get_transcript
        from pipeline.stage02_cleaner import clean_transcript
        from pipeline.stage03_chunker import chunk_transcript
        from vector_store.indexer import upsert_chunks, delete_video, video_is_indexed
        from utils.embedder import embed_texts

        client = get_qdrant()
        bm25   = get_bm25()
        cache  = get_cache_manager()

        _jobs[job_id]["step"] = "transcript"
        transcript = get_transcript(url, force_whisper=force_whisper)

        if not force_reindex and video_is_indexed(client, transcript.video_id):
            _jobs[job_id] = {
                "status": "done", "video_id": transcript.video_id,
                "video_title": transcript.video_title, "channel": transcript.channel,
                "chunks_indexed": 0, "already_existed": True,
            }
            return

        if force_reindex:
            delete_video(client, bm25, transcript.video_id)

        _jobs[job_id]["step"] = "cleaning"
        cleaned = clean_transcript(transcript)

        _jobs[job_id]["step"] = "chunking"
        chunks = chunk_transcript(cleaned)
        if not chunks:
            raise ValueError("No chunks produced — transcript may be empty.")

        _jobs[job_id]["step"] = "embedding"
        texts = [c.chunk_text for c in chunks]
        embeddings = cache.embed(texts, embed_fn=embed_texts)

        _jobs[job_id]["step"] = "indexing"
        upsert_chunks(client, bm25, chunks, embeddings, save_bm25=True)

        _jobs[job_id] = {
            "status": "done", "video_id": transcript.video_id,
            "video_title": transcript.video_title, "channel": transcript.channel,
            "chunks_indexed": len(chunks), "already_existed": False,
        }
        logger.info(f"Job {job_id}: indexed '{transcript.video_title}' ({len(chunks)} chunks)")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id] = {"status": "failed", "error": str(e)}


@router.post("/index", tags=["Indexing"])
async def index_video(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    sync: bool = Query(False, description="Wait for completion before responding"),
    _: None = Depends(verify_api_key),
):
    """
    Index a YouTube video.

    By default returns immediately with a job_id — poll /index/status/{job_id}.
    Pass ?sync=true to wait for completion (useful for testing, slow for long videos).
    """
    job_id = uuid.uuid4().hex[:8]

    if sync:
        _run_pipeline(job_id, request.url, request.force_reindex, request.force_whisper)
        job = _jobs.get(job_id, {})
        if job.get("status") == "failed":
            raise HTTPException(status_code=500, detail=job.get("error"))
        return IndexResponse(
            video_id=job["video_id"],
            video_title=job["video_title"],
            channel=job["channel"],
            chunks_indexed=job["chunks_indexed"],
            already_existed=job["already_existed"],
            message="Indexed." if not job["already_existed"] else "Already indexed.",
        )

    background_tasks.add_task(
        _run_pipeline, job_id, request.url, request.force_reindex, request.force_whisper
    )
    return {"job_id": job_id, "status": "queued", "poll": f"/index/status/{job_id}"}


@router.get("/index/status/{job_id}", tags=["Indexing"])
async def index_status(job_id: str, _: None = Depends(verify_api_key)):
    """Poll an indexing job's status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job