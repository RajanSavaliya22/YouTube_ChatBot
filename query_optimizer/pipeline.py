"""
Query Optimizer Pipeline
=========================
Orchestrates all 5 query optimization techniques into a single call.

Pipeline flow:
  Raw query
      │
      ├─► [T1] Rewrite          → retrieval-friendly phrasing
      │
      ├─► [T2] HyDE             → hypothetical doc embedding (replaces query vector)
      │
      ├─► [T3] Multi-query      → 4 paraphrases for parallel retrieval
      │
      ├─► [T4] Decompose        → sub-questions for complex queries
      │
      └─► [T5] Metadata extract → channel/date filters for Qdrant scoping
              │
              ▼
      OptimizedQuery
        .queries:      list[str]       — all query variants to retrieve for
        .vectors:      list[vector]    — one embedding per query variant
        .filters:      QueryFilters    — Qdrant payload filters
        .hyde_vector:  vector | None   — HyDE mean vector (used instead of query vec)
        .rewritten:    str             — rewritten query (for logging)

Usage:
    from query_optimizer.pipeline import QueryOptimizer

    optimizer = QueryOptimizer()
    optimized = optimizer.run("how does attention work in transformers?")

    # Use in retrieval:
    for query, vec in zip(optimized.queries, optimized.vectors):
        results = hybrid_search(..., query_vector=vec, query_text=query,
                                **optimized.filters.to_dict())
"""

from dataclasses import dataclass, field
import numpy as np

from query_optimizer.rewriter import rewrite_query
from query_optimizer.hyde import generate_hyde_vector
from query_optimizer.multiquery import expand_query
from query_optimizer.decomposer import decompose_query
from query_optimizer.metadata import extract_filters, QueryFilters
from utils.embedder import embed_query, embed_texts
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("query_optimizer.pipeline")


@dataclass
class OptimizedQuery:
    """Output of the query optimization pipeline."""
    original: str                          # Raw user input
    rewritten: str                         # After T1 rewrite
    queries: list[str]                     # All variants (multi-query + decomposed)
    vectors: list[list[float]]             # One embedding per query variant
    hyde_vector: list[float] | None        # HyDE vector (None if LLM unavailable)
    filters: QueryFilters                  # Extracted metadata filters

    @property
    def primary_vector(self) -> list[float]:
        """
        The best single vector to use for the main dense search.
        Prefers HyDE vector over plain query embedding when available.
        """
        return self.hyde_vector if self.hyde_vector is not None else self.vectors[0]

    def summary(self) -> str:
        lines = [
            f"Original:  {self.original}",
            f"Rewritten: {self.rewritten}",
            f"Queries:   {len(self.queries)} total",
        ]
        for i, q in enumerate(self.queries):
            lines.append(f"  [{i+1}] {q}")
        lines.append(f"HyDE:      {'yes' if self.hyde_vector else 'no (LLM unavailable)'}")
        lines.append(f"Filters:   {self.filters}")
        return "\n".join(lines)


class QueryOptimizer:
    """
    Runs all query optimization techniques and returns an OptimizedQuery.
    Each technique degrades gracefully if the LLM is unavailable.
    """

    @timed("stage7.query_optimizer")
    def run(
        self,
        query: str,
        explicit_channel: str | None = None,
        explicit_video_id: str | None = None,
    ) -> OptimizedQuery:
        """
        Run the full query optimization pipeline.

        Args:
            query:             Raw user query
            explicit_channel:  From CLI --channel flag
            explicit_video_id: From CLI --video flag

        Returns:
            OptimizedQuery with all variants, vectors, and filters populated
        """
        logger.info(f"Optimizing query: '{query}'")

        # ── T1: Rewrite ───────────────────────────────────────────
        rewritten = rewrite_query(query)

        # ── T2: HyDE ──────────────────────────────────────────────
        # Run on the rewritten query for best results
        hyde_vector = generate_hyde_vector(rewritten)

        # ── T3: Multi-query expansion ─────────────────────────────
        # Expand the rewritten query into paraphrases
        multi_queries = expand_query(rewritten)

        # ── T4: Sub-question decomposition ────────────────────────
        # Decompose the original (not rewritten) to preserve intent
        sub_questions = decompose_query(query)

        # Merge all query variants, deduplicate, preserve order
        all_queries = _merge_unique([rewritten] + multi_queries + sub_questions)

        # ── T5: Metadata filter extraction ────────────────────────
        filters = extract_filters(
            query,
            explicit_channel=explicit_channel,
            explicit_video_id=explicit_video_id,
        )

        # ── Embed all query variants ──────────────────────────────
        vectors = _embed_all(all_queries)

        optimized = OptimizedQuery(
            original=query,
            rewritten=rewritten,
            queries=all_queries,
            vectors=vectors,
            hyde_vector=hyde_vector,
            filters=filters,
        )

        logger.info(f"\n{optimized.summary()}")
        return optimized


# ── Helpers ───────────────────────────────────────────────────

def _merge_unique(queries: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(q)
    return result


def _embed_all(queries: list[str]) -> list[list[float]]:
    """
    Embed all query variants.
    Uses batch embedding for efficiency (single model forward pass).
    """
    if not queries:
        return []

    # BGE queries need the retrieval prefix
    from config import EMBEDDING
    prefixed = [f"{EMBEDDING.query_prefix}{q}" for q in queries]

    from sentence_transformers import SentenceTransformer
    from utils.embedder import get_model
    model = get_model()
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()