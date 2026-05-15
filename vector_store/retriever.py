"""
Stage 5c: Retriever
====================
Hybrid retrieval combining Qdrant dense search + BM25 sparse search,
fused via Reciprocal Rank Fusion (RRF).

Search pipeline:
  1. Dense ANN search  → Qdrant (semantic similarity)
  2. Sparse BM25 search → in-memory index (keyword matching)
  3. RRF fusion        → unified ranking
  4. Results returned  → ready for re-ranking (Stage 6)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchParams,
    QuantizationSearchParams,
)

from schema import ChunkPayload, RetrievedChunk
from vector_store.sparse.bm25_store import BM25Store
from vector_store.fusion import reciprocal_rank_fusion, FusedResult
from config import COLLECTION, FUSION
from utils.logger import get_logger

logger = get_logger("vector_store.retriever")


def hybrid_search(
    client: QdrantClient,
    bm25_store: BM25Store,
    query_vector: list[float],
    query_text: str,
    top_k: int = 20,
    score_threshold: float = 0.35,
    filter_video_id: str | None = None,
    filter_channel: str | None = None,
    filter_date_after: str | None = None,
    filter_language: str = "en",
) -> list[FusedResult]:
    """
    Full hybrid retrieval: dense + sparse + RRF.

    Args:
        client:           Active QdrantClient
        bm25_store:       Loaded BM25Store instance
        query_vector:     Embedded query vector (from utils.embedder.embed_query)
        query_text:       Raw query string (for BM25 tokenization)
        top_k:            Final result count after fusion
        score_threshold:  Minimum cosine score for dense results
        filter_video_id:  Scope search to one video
        filter_channel:   Scope search to one channel
        filter_date_after: Only return chunks from videos after this date (YYYYMMDD)
        filter_language:  Language code (default "en")

    Returns:
        List of FusedResult sorted by RRF score descending
    """

    # ── 1. DENSE SEARCH (Qdrant ANN) ─────────────────────────────
    must_conditions = [
        FieldCondition(key="language", match=MatchValue(value=filter_language))
    ]

    if filter_video_id:
        must_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=filter_video_id))
        )
    if filter_channel:
        must_conditions.append(
            FieldCondition(key="channel", match=MatchValue(value=filter_channel))
        )
    if filter_date_after:
        must_conditions.append(
            FieldCondition(key="published_date", range=Range(gte=filter_date_after))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    dense_hits = client.query_points(
        collection_name=COLLECTION.name,
        query=query_vector,
        limit=FUSION.dense_top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        search_params=SearchParams(
            hnsw_ef=128,            # Candidates explored during search (higher = more accurate)
            exact=False,            # Use HNSW, not brute-force
            quantization=QuantizationSearchParams(
                ignore=False,
                rescore=True,       # Re-score top candidates with full-precision vectors
                oversampling=2.0,   # Fetch 2× more candidates before rescoring
            ),
        ),
        with_payload=True,
        with_vectors=False,         # Don't return vectors — saves bandwidth
    )

    dense_results = [
        RetrievedChunk(
            payload=ChunkPayload(**hit.payload),
            score=hit.score,
            chunk_id=str(hit.id),
        )
        for hit in dense_hits.points
    ]
    logger.info(f"Dense search: {len(dense_results)} results (threshold={score_threshold})")

    # ── 2. SPARSE SEARCH (BM25) ───────────────────────────────────
    sparse_results = bm25_store.search(
        query=query_text,
        top_k=FUSION.sparse_top_k,
        filter_video_id=filter_video_id,
        filter_channel=filter_channel,
    )
    logger.info(f"Sparse search: {len(sparse_results)} results")

    # ── 3. RRF FUSION ─────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=top_k,
    )

    return fused


def dense_only_search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 20,
    score_threshold: float = 0.4,
    filter_video_id: str | None = None,
    filter_channel: str | None = None,
) -> list[RetrievedChunk]:
    """
    Dense-only fallback search (used when BM25 store is unavailable
    or during initial indexing before BM25 is populated).
    """
    must_conditions = []
    if filter_video_id:
        must_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=filter_video_id))
        )
    if filter_channel:
        must_conditions.append(
            FieldCondition(key="channel", match=MatchValue(value=filter_channel))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    hits = client.search(
        collection_name=COLLECTION.name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    return [
        RetrievedChunk(
            payload=ChunkPayload(**hit.payload),
            score=hit.score,
            chunk_id=str(hit.id),
        )
        for hit in hits
    ]


def search_by_video_ids(
    client: QdrantClient,
    query_vector: list[float],
    video_ids: list[str],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Search within a specific set of videos (multi-video scoped search)."""
    hits = client.search(
        collection_name=COLLECTION.name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchAny(any=video_ids))]
        ),
        with_payload=True,
        with_vectors=False,
    )

    return [
        RetrievedChunk(
            payload=ChunkPayload(**hit.payload),
            score=hit.score,
            chunk_id=str(hit.id),
        )
        for hit in hits
    ]
