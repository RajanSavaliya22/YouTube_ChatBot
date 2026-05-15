"""
L1 — Exact Query Cache
========================
Caches full pipeline answers keyed by a normalized query hash.

Key:   rag:exact:{sha256(normalized_query)}
Value: JSON-serialized answer (retrieved chunks + metadata)
TTL:   1 hour (configurable via EXACT_CACHE_TTL)

Best for: repeated identical or near-identical queries (e.g. a popular
question asked by many users). Zero retrieval cost on hit.

Normalization before hashing:
  - lowercase
  - collapse whitespace
  - strip leading/trailing punctuation
This means "What is RAG?" and "what is rag" hash to the same key.
"""

import hashlib
import json
import re
from typing import Any

from cache.redis_client import get_redis
from config import CACHE
from utils.logger import get_logger

logger = get_logger("cache.exact")


def _normalize(query: str) -> str:
    """Normalize query for consistent cache key generation."""
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)                 # Collapse whitespace
    q = re.sub(r"^[^\w]+|[^\w]+$", "", q)      # Strip leading/trailing punctuation
    return q


def _make_key(query: str, **filter_kwargs) -> str:
    """
    Build a Redis key from the normalized query + any active filters.
    Filters (video_id, channel) are part of the key so scoped queries
    don't return results from a broader cached query.
    """
    normalized = _normalize(query)
    # Append sorted filter k=v pairs so order doesn't matter
    filter_str = "&".join(f"{k}={v}" for k, v in sorted(filter_kwargs.items()) if v)
    raw = f"{normalized}|{filter_str}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{CACHE.exact_prefix}{digest}"


# ── Public API ────────────────────────────────────────────────

def get(query: str, **filter_kwargs) -> Any | None:
    """
    Retrieve a cached answer for this exact query + filters.
    Returns deserialized Python object, or None on miss.
    """
    if not CACHE.enabled:
        return None
    r = get_redis()
    if r is None:
        return None

    key = _make_key(query, **filter_kwargs)
    raw = r.get(key)
    if raw is None:
        logger.debug(f"L1 MISS: {query[:60]}")
        return None

    logger.debug(f"L1 HIT:  {query[:60]}")
    return json.loads(raw)


def set(query: str, answer: Any, **filter_kwargs) -> None:
    """
    Store an answer in the exact cache.
    `answer` must be JSON-serializable.
    """
    if not CACHE.enabled:
        return
    r = get_redis()
    if r is None:
        return

    key = _make_key(query, **filter_kwargs)
    r.setex(key, CACHE.exact_ttl, json.dumps(answer, ensure_ascii=False))
    logger.debug(f"L1 SET:  {query[:60]} (ttl={CACHE.exact_ttl}s)")


def invalidate(query: str, **filter_kwargs) -> None:
    """Remove a specific query from the exact cache (e.g. after re-indexing)."""
    r = get_redis()
    if r is None:
        return
    key = _make_key(query, **filter_kwargs)
    r.delete(key)