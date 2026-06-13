"""
Stage 8: Re-ranker
===================
Supports two backends, switched via RERANKER_BACKEND env var:

  local  (default) — BAAI/bge-reranker-base loaded in-process (~500MB RAM)
  voyage             — Voyage AI API reranking (no local model, zero RAM)

Use 'voyage' backend for Render free tier deployment.
Use 'local' for local development.

Voyage reranking model: "rerank-2.5"
  - Scores (query, passage) pairs via API
  - Returns relevance_score: 0.0–1.0 (normalized, unlike local logits)
  - Free tier: 100 rerank requests/minute
  - No local RAM cost
"""

import os
from dataclasses import dataclass
from config import RERANKER
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("reranker.model")

BACKEND = os.getenv("RERANKER_BACKEND", "local")  # "local" | "voyage"
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
_voyage_client = None

# ── Local model singleton ─────────────────────────────────────

_local_model = None


def get_model():
    """Load local cross-encoder model once (only used in local backend)."""
    global _local_model
    if _local_model is not None:
        return _local_model

    if BACKEND == "voyage":
        return None  # Not needed for Voyage backend

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading re-ranker: {RERANKER.model_name} on {RERANKER.device}")
        _local_model = CrossEncoder(
            RERANKER.model_name,
            device=RERANKER.device,
            max_length=512,
        )
        logger.info("Re-ranker model loaded.")
        return _local_model
    except Exception as e:
        logger.warning(f"Re-ranker model failed to load: {e}")
        return None



def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        _voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        logger.info(f"Voyage reranker client ready (model={VOYAGE_RERANK_MODEL})")
    return _voyage_client

def _rerank_voyage(query: str, candidates: list, top_n: int) -> list["RankedChunk"]:
    client = _get_voyage_client()
    passages = [c.payload.parent_text for c in candidates]

    logger.info(f"Voyage reranking {len(passages)} candidates for: '{query[:60]}'")

    result = client.rerank(
        query=query,
        documents=passages,
        model=VOYAGE_RERANK_MODEL,
        top_k=top_n,
        truncation=True,   # Auto-truncate to 32K token limit
    )

    ranked = []
    for item in result.results:
        candidate = candidates[item.index]
        if item.relevance_score < RERANKER.score_threshold:
            continue
        ranked.append(RankedChunk(
            chunk_id=candidate.chunk_id,
            payload=candidate.payload,
            rerank_score=item.relevance_score,
            retrieval_rrf=candidate.rrf_score,
            retrieval_rank=item.index + 1,
        ))

    _log_ranking_changes(ranked)
    return ranked


def _rerank_local(
    query: str,
    candidates: list,
    top_n: int,
) -> list["RankedChunk"]:
    """Rerank using local cross-encoder model."""
    model = get_model()
    if model is None:
        logger.warning("Re-ranker unavailable — falling back to retrieval order.")
        return _fallback(candidates, top_n)

    pairs = [(query, c.payload.parent_text) for c in candidates]
    logger.info(f"Local reranking {len(pairs)} candidates for: '{query[:60]}'")

    scores = model.predict(
        pairs,
        batch_size=RERANKER.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    ranked = []
    for original_rank, (candidate, score) in enumerate(zip(candidates, scores), start=1):
        score = float(score)
        if score < RERANKER.score_threshold:
            continue
        ranked.append(RankedChunk(
            chunk_id=candidate.chunk_id,
            payload=candidate.payload,
            rerank_score=score,
            retrieval_rrf=candidate.rrf_score,
            retrieval_rank=original_rank,
        ))

    ranked.sort(key=lambda x: x.rerank_score, reverse=True)
    final = ranked[:top_n]
    _log_ranking_changes(final)
    return final


# ── Data class ────────────────────────────────────────────────

@dataclass
class RankedChunk:
    chunk_id: str
    payload: object
    rerank_score: float
    retrieval_rrf: float
    retrieval_rank: int

    @property
    def timestamp_url(self) -> str:
        p = self.payload
        return f"{p.video_url}?t={int(p.timestamp_start)}"


# ── Public API ────────────────────────────────────────────────

@timed("stage8.reranker")
def rerank(
    query: str,
    candidates: list,
    top_n: int | None = None,
) -> list[RankedChunk]:
    """
    Re-rank retrieval candidates.
    Routes to Voyage or local backend based on RERANKER_BACKEND env var.
    """
    top_n = top_n or RERANKER.top_n
    if not candidates:
        return []
    if not RERANKER.enabled:
        return _fallback(candidates, top_n)
    if BACKEND == "voyage":
        return _rerank_voyage(query, candidates, top_n)
    return _rerank_local(query, candidates, top_n)


def _fallback(candidates: list, top_n: int) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=c.chunk_id,
            payload=c.payload,
            rerank_score=c.rrf_score,
            retrieval_rrf=c.rrf_score,
            retrieval_rank=i + 1,
        )
        for i, c in enumerate(candidates[:top_n])
    ]


def _log_ranking_changes(ranked: list[RankedChunk]) -> None:
    logger.info(f"Re-ranked top {len(ranked)} results:")
    for new_rank, chunk in enumerate(ranked, start=1):
        movement = chunk.retrieval_rank - new_rank
        arrow = f"↑{movement}" if movement > 0 else (f"↓{abs(movement)}" if movement < 0 else "=")
        logger.info(
            f"  #{new_rank} [{arrow}] score={chunk.rerank_score:.4f} "
            f"(was #{chunk.retrieval_rank}) | {chunk.payload.chunk_text[:70]}..."
        )