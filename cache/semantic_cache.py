"""
L2 — Semantic Query Cache
===========================
Caches answers keyed by query embedding similarity, not exact text match.
A new query hits the cache if a past query's embedding is within cosine
similarity threshold (default 0.95) of the new query's embedding.

Storage layout in Redis:
  rag:sem:index   →  JSON list of {key, vector, query} entries (the index)
  rag:sem:{uuid}  →  JSON answer payload for that entry

Why not use Qdrant for this?
  Keeping semantic cache in Redis avoids a round-trip to Qdrant during the
  cache-check phase and keeps the cache self-contained. For very large
  deployments (>100k cached queries) you'd move the vector index to Qdrant.

Eviction: LRU-style — when max_entries is reached, the oldest entry is dropped.
"""

import json
import math
import uuid
from typing import Any

from cache.redis_client import get_redis
from config import CACHE
from utils.logger import get_logger

logger = get_logger("cache.semantic")

_INDEX_KEY = f"{CACHE.semantic_prefix}index"


# ── Math ──────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors."""
    # BGE embeddings are L2-normalized at embed time, so dot product = cosine sim
    return sum(x * y for x, y in zip(a, b))


# ── Index helpers ─────────────────────────────────────────────

def _load_index(r) -> list[dict]:
    """Load the semantic index from Redis. Returns [] if empty."""
    raw = r.get(_INDEX_KEY)
    if raw is None:
        return []
    return json.loads(raw)


def _save_index(r, index: list[dict]) -> None:
    """Persist the semantic index to Redis (no TTL — managed manually)."""
    r.set(_INDEX_KEY, json.dumps(index))


# ── Public API ────────────────────────────────────────────────

def get(query_vector: list[float], **filter_kwargs) -> Any | None:
    """
    Check if any cached query is semantically similar to this query vector,
    restricted to entries with matching filters (e.g. same video_id/channel).

    Without filter matching, two different videos asked the same general
    question ("what is this video about?") would collide — the embedding
    of the question text is nearly identical regardless of which video it
    refers to, so the filter check is what keeps answers scoped correctly.

    Steps:
      1. Load the in-Redis index of (entry_key, vector, filters) tuples
      2. Discard entries whose filters don't match this request
      3. Compute cosine similarity between query_vector and each remaining vector
      4. If best match ≥ threshold, fetch and return its cached answer

    Returns deserialized answer dict, or None on miss.
    """
    if not CACHE.enabled:
        return None
    r = get_redis()
    if r is None:
        return None

    index = _load_index(r)
    if not index:
        return None

    best_score = -1.0
    best_entry = None

    for entry in index:
        # Skip entries scoped to a different video/channel/etc.
        # Entries written before this fix have no "filters" key — treat
        # missing filters as {} so old global-scope entries still compare
        # equal only against other filter-less lookups (back-compat).
        if entry.get("filters", {}) != filter_kwargs:
            continue

        score = _cosine_similarity(query_vector, entry["vector"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= CACHE.semantic_threshold and best_entry:
        answer_raw = r.get(best_entry["key"])
        if answer_raw:
            logger.debug(
                f"L2 HIT:  sim={best_score:.4f} filters={filter_kwargs} "
                f"matched='{best_entry['query'][:50]}'"
            )
            # Refresh TTL on hit (promotes hot entries)
            r.expire(best_entry["key"], CACHE.semantic_ttl)
            return json.loads(answer_raw)

    logger.debug(
        f"L2 MISS: best_sim={best_score:.4f} filters={filter_kwargs} "
        f"(threshold={CACHE.semantic_threshold})"
    )
    return None


def set(query: str, query_vector: list[float], answer: Any, **filter_kwargs) -> None:
    """
    Store an answer in the semantic cache, scoped to the given filters
    (e.g. video_id/channel) so it can only be matched by future lookups
    with the same filters.

    Steps:
      1. Generate a unique key for this entry
      2. Store the answer JSON under that key with TTL
      3. Append (key, vector, query, filters) to the index
      4. Evict oldest entry if index exceeds max_entries
    """
    if not CACHE.enabled:
        return
    r = get_redis()
    if r is None:
        return

    entry_key = f"{CACHE.semantic_prefix}{uuid.uuid4().hex}"
    r.setex(entry_key, CACHE.semantic_ttl, json.dumps(answer, ensure_ascii=False))

    index = _load_index(r)

    # Evict oldest if at capacity (simple FIFO eviction)
    if len(index) >= CACHE.semantic_max_entries:
        oldest = index.pop(0)
        r.delete(oldest["key"])
        logger.debug(f"L2 evicted oldest entry: {oldest['query'][:40]}")

    index.append({
        "key": entry_key,
        "vector": query_vector,
        "query": query[:100],   # Store truncated query for debug logging only
        "filters": filter_kwargs,
    })
    _save_index(r, index)

    logger.debug(f"L2 SET:  '{query[:60]}' filters={filter_kwargs} (index_size={len(index)})")


def get_stats(r=None) -> dict:
    """Return semantic cache statistics."""
    if r is None:
        r = get_redis()
    if r is None:
        return {"available": False}
    index = _load_index(r)
    return {
        "available": True,
        "entries": len(index),
        "max_entries": CACHE.semantic_max_entries,
        "threshold": CACHE.semantic_threshold,
    }