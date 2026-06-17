"""
Cache Manager
==============
Single entry point for all caching logic.
All pipeline and query code imports from here — never from individual cache modules.

Two-level query cache lookup order:
  L1 (Exact)    →  hash of normalized query text      ~0ms hit
  L2 (Semantic) →  cosine similarity of query vector  ~5ms hit
  Miss          →  run full retrieval pipeline

L3 (Embedding) is invoked transparently inside embed_with_cache().

Usage in query path:
    from cache.manager import CacheManager
    cache = CacheManager()

    result = cache.get_query(query, query_vector)
    if result is None:
        result = run_full_pipeline(query)
        cache.set_query(query, query_vector, result)

Usage in embedding path:
    vectors = cache.embed(texts, embed_fn=embed_texts)
"""

from typing import Any, Callable

from cache import exact_cache, semantic_cache, embedding_cache
from cache.redis_client import get_redis, is_available, flush_all_rag_keys
from utils.logger import get_logger

logger = get_logger("cache.manager")


class CacheManager:
    """
    Unified cache interface.

    Query cache (L1 + L2):
      - get_query()  → check L1 then L2
      - set_query()  → write to both L1 and L2

    Embedding cache (L3):
      - embed()      → embed_with_cache() wrapper
    """

    # ── Query cache ───────────────────────────────────────────────

    def get_query(
        self,
        query: str,
        query_vector: list[float],
        **filter_kwargs,
    ) -> Any | None:
        """
        Look up a query through both cache layers.

        Returns cached answer on hit, None on miss.
        Logs which layer served the hit.
        """
        # L1 — Exact match (cheapest: just a Redis GET by hash key)
        result = exact_cache.get(query, **filter_kwargs)
        if result is not None:
            logger.info(f"Cache HIT  [L1-Exact]    '{query[:60]}'")
            return result

        # L2 — Semantic match (Redis GET index + cosine similarity scan)
        # filter_kwargs (video_id, channel) scopes the match so the same
        # question asked about different videos doesn't collide.
        result = semantic_cache.get(query_vector, **filter_kwargs)
        if result is not None:
            logger.info(f"Cache HIT  [L2-Semantic] '{query[:60]}'")
            # Promote to L1 so next identical query is O(1)
            exact_cache.set(query, result, **filter_kwargs)
            return result

        logger.info(f"Cache MISS             '{query[:60]}'")
        return None

    def set_query(
        self,
        query: str,
        query_vector: list[float],
        answer: Any,
        **filter_kwargs,
    ) -> None:
        """
        Write an answer to both L1 (exact) and L2 (semantic) caches.
        Called after a successful pipeline run.
        """
        exact_cache.set(query, answer, **filter_kwargs)
        semantic_cache.set(query, query_vector, answer, **filter_kwargs)
        logger.debug(f"Cache SET  [L1+L2]      '{query[:60]}'")

    def invalidate_query(self, query: str, **filter_kwargs) -> None:
        """Remove a specific query from L1 cache (L2 expires by TTL)."""
        exact_cache.invalidate(query, **filter_kwargs)

    # ── Embedding cache ───────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> list[list[float]]:
        """
        Embed texts using L3 cache.
        Only texts not in cache are passed to embed_fn.
        """
        return embedding_cache.embed_with_cache(texts, embed_fn)

    # ── Admin / stats ─────────────────────────────────────────────

    def stats(self) -> dict:
        """Return stats for all cache layers."""
        available = is_available()
        if not available:
            return {"available": False, "message": "Redis not connected"}

        r = get_redis()
        sem_stats = semantic_cache.get_stats(r)

        # Count keys per layer
        def count_keys(prefix: str) -> int:
            keys = r.keys(f"{prefix}*")
            return len(keys)

        from config import CACHE
        return {
            "available": True,
            "L1_exact": {
                "keys": count_keys(CACHE.exact_prefix),
                "ttl_seconds": CACHE.exact_ttl,
            },
            "L2_semantic": {
                **sem_stats,
                "ttl_seconds": CACHE.semantic_ttl,
            },
            "L3_embedding": {
                "keys": count_keys(CACHE.embed_prefix),
                "ttl_seconds": CACHE.embed_ttl,
            },
        }

    def flush(self) -> int:
        """Flush all RAG cache keys from Redis. Returns count deleted."""
        return flush_all_rag_keys()