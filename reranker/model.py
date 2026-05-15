"""
Stage 8: Re-ranker
===================
Uses a cross-encoder model to re-score the top candidates from hybrid
retrieval. Unlike the bi-encoder (which embeds query and chunk separately),
the cross-encoder reads (query + chunk) together — far more accurate.

Model: BAAI/bge-reranker-large
  - Cross-encoder architecture (BERT-based)
  - Input: [query, passage] pair
  - Output: single relevance score (logit, not bounded)
  - ~560MB, runs on CPU in ~50–200ms for 20 candidates

Pipeline position:
  Stage 7 retrieval → top 20 fused chunks
       │
       ▼
  Re-ranker scores all 20 (query, chunk) pairs     ← here
       │
       ▼
  Top 5 by re-rank score → Stage 9 LLM generation

Why cross-encoder beats bi-encoder for final ranking:
  Bi-encoder: embed(query) · embed(chunk)  — fast, approximate
  Cross-encoder: score(query ⊕ chunk)      — slow, accurate
  We use bi-encoder for recall (top 20) and cross-encoder for precision (top 5).

Fallback: if model unavailable, returns input list truncated to top_n.
"""

from dataclasses import dataclass
from config import RERANKER
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("reranker.model")

_model = None   # Module-level singleton


def get_model():
    """Load reranker model once and cache in memory."""
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading re-ranker: {RERANKER.model_name} on {RERANKER.device}")
        _model = CrossEncoder(
            RERANKER.model_name,
            device=RERANKER.device,
            cache_folder=RERANKER.cache_folder,
            max_length=512,   # Truncate (query + chunk) to 512 tokens
        )
        logger.info("Re-ranker model loaded.")
        return _model
    except Exception as e:
        logger.warning(f"Re-ranker model failed to load: {e}")
        return None


@dataclass
class RankedChunk:
    """A single chunk after re-ranking, with both retrieval and rerank scores."""
    chunk_id: str
    payload: object          # ChunkPayload
    rerank_score: float      # Cross-encoder score (higher = more relevant)
    retrieval_rrf: float     # Original RRF score from Stage 7
    retrieval_rank: int      # Original rank before reranking (1-indexed)

    @property
    def timestamp_url(self) -> str:
        p = self.payload
        return f"{p.video_url}?t={int(p.timestamp_start)}"


@timed("stage8.reranker")
def rerank(
    query: str,
    candidates: list,        # list[FusedResult] from Stage 7
    top_n: int | None = None,
) -> list[RankedChunk]:
    """
    Re-rank retrieval candidates using a cross-encoder.

    Args:
        query:      The user's query (or rewritten query from Stage 7)
        candidates: FusedResult list from multi_query_retrieve()
        top_n:      How many to keep (defaults to RERANKER.top_n)

    Returns:
        List of RankedChunk sorted by rerank_score descending,
        truncated to top_n. Falls back to retrieval order if model unavailable.
    """
    top_n = top_n or RERANKER.top_n

    if not candidates:
        return []

    if not RERANKER.enabled:
        logger.info("Re-ranker disabled — returning top_n by retrieval score.")
        return _fallback(candidates, top_n)

    model = get_model()
    if model is None:
        logger.warning("Re-ranker unavailable — falling back to retrieval order.")
        return _fallback(candidates, top_n)

    # Build (query, passage) pairs for the cross-encoder
    # Use parent_text (richer context) not chunk_text (the small embedded unit)
    pairs = [
        (query, candidate.payload.parent_text)
        for candidate in candidates
    ]

    logger.info(f"Re-ranking {len(pairs)} candidates for: '{query[:60]}'")

    # Score all pairs in one batch
    scores = model.predict(
        pairs,
        batch_size=RERANKER.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Build RankedChunk list
    ranked = []
    for original_rank, (candidate, score) in enumerate(zip(candidates, scores), start=1):
        score = float(score)
        if score < RERANKER.score_threshold:
            logger.debug(
                f"  Dropped (score={score:.3f} < threshold={RERANKER.score_threshold}): "
                f"{candidate.payload.chunk_text[:60]}"
            )
            continue
        ranked.append(RankedChunk(
            chunk_id=candidate.chunk_id,
            payload=candidate.payload,
            rerank_score=score,
            retrieval_rrf=candidate.rrf_score,
            retrieval_rank=original_rank,
        ))

    # Sort by cross-encoder score descending
    ranked.sort(key=lambda x: x.rerank_score, reverse=True)

    final = ranked[:top_n]

    # Log ranking changes (useful for debugging)
    _log_ranking_changes(final)

    return final


def _fallback(candidates: list, top_n: int) -> list[RankedChunk]:
    """Return top_n candidates in retrieval order (no reranking)."""
    return [
        RankedChunk(
            chunk_id=c.chunk_id,
            payload=c.payload,
            rerank_score=c.rrf_score,   # Use RRF score as proxy
            retrieval_rrf=c.rrf_score,
            retrieval_rank=i + 1,
        )
        for i, c in enumerate(candidates[:top_n])
    ]


def _log_ranking_changes(ranked: list[RankedChunk]) -> None:
    """Log how re-ranking changed the order vs retrieval."""
    logger.info(f"Re-ranked top {len(ranked)} results:")
    for new_rank, chunk in enumerate(ranked, start=1):
        movement = chunk.retrieval_rank - new_rank
        arrow = (
            f"↑{movement}"  if movement > 0
            else f"↓{abs(movement)}" if movement < 0
            else "="
        )
        logger.info(
            f"  #{new_rank} [{arrow}] score={chunk.rerank_score:.3f} "
            f"(was #{chunk.retrieval_rank}) | {chunk.payload.chunk_text[:70]}..."
        )