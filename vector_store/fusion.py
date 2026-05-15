"""
Reciprocal Rank Fusion (RRF)
=============================
Merges dense (Qdrant) and sparse (BM25) result lists into a single
unified ranking without requiring any additional model or scoring.

RRF formula:  RRF(d) = Σ 1 / (k + rank(d))
  k = 60      (smoothing constant — research-validated default)
  rank(d)     = 1-indexed position in a result list

Documents appearing in both lists receive scores from both terms,
naturally boosting results relevant by both semantic and keyword signal.

Reference: Cormack, Clarke & Buettcher (2009) — SIGIR
"""

import logging
from dataclasses import dataclass

from schema import ChunkPayload, RetrievedChunk
from config import FUSION
from utils.logger import get_logger

logger = get_logger("vector_store.fusion")


@dataclass
class FusedResult:
    """A single result after RRF fusion, carrying both search signals."""
    chunk_id:     str
    payload:      ChunkPayload
    rrf_score:    float
    dense_rank:   int | None    # 1-indexed rank in dense results (None if absent)
    sparse_rank:  int | None    # 1-indexed rank in BM25 results (None if absent)
    dense_score:  float | None  # Original cosine similarity score
    sparse_score: float | None  # Original BM25 score

    @property
    def found_in_both(self) -> bool:
        """True when this result appeared in both dense and sparse results."""
        return self.dense_rank is not None and self.sparse_rank is not None

    @property
    def source(self) -> str:
        if self.found_in_both:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        return "sparse"


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[tuple[ChunkPayload, float, str]],
    top_k: int | None = None,
) -> list[FusedResult]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion.

    Args:
        dense_results:  List of RetrievedChunk from Qdrant ANN search
        sparse_results: List of (ChunkPayload, score, chunk_id) from BM25
        top_k:          Max results to return (defaults to FUSION.final_top_k)

    Returns:
        List of FusedResult sorted by RRF score descending
    """
    top_k = top_k or FUSION.final_top_k
    k = FUSION.rrf_k

    # Build lookup maps: chunk_id → (payload, score, rank)
    dense_map: dict[str, tuple[ChunkPayload, float, int]] = {
        r.chunk_id: (r.payload, r.score, i + 1)
        for i, r in enumerate(dense_results)
    }
    sparse_map: dict[str, tuple[ChunkPayload, float, int]] = {
        chunk_id: (payload, score, i + 1)
        for i, (payload, score, chunk_id) in enumerate(sparse_results)
    }

    # Union of all seen chunk IDs
    all_ids = set(dense_map) | set(sparse_map)

    fused: list[FusedResult] = []

    for chunk_id in all_ids:
        rrf_score    = 0.0
        dense_rank   = None
        sparse_rank  = None
        dense_score  = None
        sparse_score = None
        payload      = None

        if chunk_id in dense_map:
            payload, dense_score, dense_rank = dense_map[chunk_id]
            rrf_score += 1.0 / (k + dense_rank)

        if chunk_id in sparse_map:
            sparse_payload, sparse_score, sparse_rank = sparse_map[chunk_id]
            rrf_score += 1.0 / (k + sparse_rank)
            if payload is None:
                payload = sparse_payload  # Use sparse payload if dense not present

        if payload is None:
            continue

        fused.append(FusedResult(
            chunk_id=chunk_id,
            payload=payload,
            rrf_score=rrf_score,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            dense_score=dense_score,
            sparse_score=sparse_score,
        ))

    # Sort by RRF score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    top = fused[:top_k]

    # Log fusion diagnostics
    both_count   = sum(1 for r in top if r.found_in_both)
    dense_only   = sum(1 for r in top if r.source == "dense")
    sparse_only  = sum(1 for r in top if r.source == "sparse")

    logger.info(
        f"RRF: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(top)} results | "
        f"both={both_count} dense_only={dense_only} sparse_only={sparse_only}"
    )

    return top
