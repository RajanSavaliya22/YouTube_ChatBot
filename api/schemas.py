"""
API Request / Response schemas.
All endpoints share these Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── /index ────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    force_reindex: bool = Field(False, description="Re-index even if video exists")
    force_whisper: bool = Field(False, description="Force Whisper transcription")


class IndexResponse(BaseModel):
    video_id: str
    video_title: str
    channel: str
    chunks_indexed: int
    already_existed: bool
    message: str


# ── /query ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int    = Field(20,  ge=1,  le=100)
    top_n: int    = Field(5,   ge=1,  le=20)
    channel:  Optional[str] = None
    video_id: Optional[str] = None
    no_cache:    bool = False
    no_optimize: bool = False
    no_rerank:   bool = False
    force_overview:  bool = False
    force_specific:  bool = False


class CitationModel(BaseModel):
    index: int
    video_title: str
    channel: str
    timestamp_label: str
    url_with_timestamp: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    is_confident: bool
    from_cache: bool
    route: str    # "specific" | "overview"
    model: str


# ── /health ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str    # "healthy" | "degraded" | "unhealthy"
    qdrant: bool
    redis: bool
    llm_api: bool
    version: str = "1.0.0"


# ── /videos ───────────────────────────────────────────────────

class VideoItem(BaseModel):
    video_id: str
    video_title: str
    channel: str
    video_url: str
    chunk_count: int