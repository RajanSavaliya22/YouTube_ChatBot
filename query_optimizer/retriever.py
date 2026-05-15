"""
Multi-Query Retriever
======================
Runs hybrid_search for every query variant produced by the optimizer,
then fuses all result lists into a single unified ranking via RRF.

This is the retrieval entry point when Stage 7 is active.
It replaces the single hybrid_search() call in main.py with a
multi-pass retrieval that leverages all optimized query variants.

Flow:
  OptimizedQuery.queries  [q1, q2, q3, q4, q5]
       │
       ├─► hybrid_search(q1, hyde_vector or vec1)  → results_1
       ├─► hybrid_search(q2, vec2)                 → results_2
       ├─► hybrid_search(q3, vec3)                 → results_3
       │        ...
       └─► RRF fusion of all result lists          → final top_k
"""

import logging
from collections import defaultdict


# hybrid_search imported lazily inside function to avoid qdrant_client at module load

from vector_store.fusion import FusedResult
from query_optimizer.pipeline import OptimizedQuery
from config import FUSION
from utils.logger import get_logger

logger = get_logger("query_optimizer.retriever")


def multi_query_retrieve(
    client,  # QdrantClient
    bm25_store,  # BM25Store
    optimized: OptimizedQuery,
    top_k: int = 20,
    score_threshold: float = 0.35,
) -> list[FusedResult]:
    """
    Retrieve and fuse results for all query variants.

    Args:
        client:          Active QdrantClient
        bm25_store:      Loaded BM25Store
        optimized:       OptimizedQuery from Stage 7 pipeline
        top_k:           Final result count after all fusion
        score_threshold: Min cosine score for dense retrieval

    Returns:
        Final fused and deduplicated list of FusedResult
    """
    filter_kwargs = optimized.filters.to_dict()
    all_per_query_results: list[list[FusedResult]] = []

    for i, (query_text, query_vec) in enumerate(
        zip(optimized.queries, optimized.vectors)
    ):
        # First query uses HyDE vector if available (better retrieval)
        vec = optimized.hyde_vector if (i == 0 and optimized.hyde_vector) else query_vec

        logger.info(
            f"  Retrieving [{i+1}/{len(optimized.queries)}]: '{query_text[:60]}'"
            + (" [HyDE]" if vec is optimized.hyde_vector else "")
        )

        from vector_store.retriever import hybrid_search
        results = hybrid_search(
            client=client,
            bm25_store=bm25_store,
            query_vector=vec,
            query_text=query_text,
            top_k=top_k,             # Fetch top_k per query before final fusion
            score_threshold=score_threshold,
            **filter_kwargs,
        )
        all_per_query_results.append(results)

    if not all_per_query_results:
        return []

    # Single query — return directly
    if len(all_per_query_results) == 1:
        return all_per_query_results[0]

    # Multi-query RRF: fuse across all per-query result lists
    final = _fuse_multi_query_results(all_per_query_results, top_k=top_k)

    logger.info(
        f"Multi-query fusion: {len(optimized.queries)} queries "
        f"→ {sum(len(r) for r in all_per_query_results)} total candidates "
        f"→ {len(final)} final results"
    )
    return final


def _fuse_multi_query_results(
    per_query_results: list[list[FusedResult]],
    top_k: int = 20,
) -> list[FusedResult]:
    """
    Apply RRF across multiple per-query result lists.

    Each per-query result list is already RRF-fused (dense + sparse).
    This applies a second round of RRF across the lists themselves.
    Chunks appearing in multiple query result sets get boosted scores.
    """
    k = FUSION.rrf_k

    # chunk_id → accumulated RRF score across all query passes
    rrf_scores: dict[str, float] = defaultdict(float)
    # chunk_id → best FusedResult (for payload access)
    best_result: dict[str, FusedResult] = {}

    for query_results in per_query_results:
        for rank, result in enumerate(query_results, start=1):
            rrf_scores[result.chunk_id] += 1.0 / (k + rank)
            # Keep whichever pass gave the highest individual RRF score
            if result.chunk_id not in best_result or \
               result.rrf_score > best_result[result.chunk_id].rrf_score:
                best_result[result.chunk_id] = result

    # Build final list with updated cross-query RRF scores
    final = []
    for chunk_id, cross_rrf in rrf_scores.items():
        result = best_result[chunk_id]
        # Create new FusedResult with the cross-query RRF score
        final.append(FusedResult(
            chunk_id=chunk_id,
            payload=result.payload,
            rrf_score=cross_rrf,           # Cross-query RRF score
            dense_rank=result.dense_rank,
            sparse_rank=result.sparse_rank,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
        ))

    final.sort(key=lambda x: x.rrf_score, reverse=True)
    return final[:top_k]