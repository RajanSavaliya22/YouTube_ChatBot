"""
L3 — Embedding Cache
======================
Caches embedding vectors in Redis to avoid re-computing them for
chunks that have already been embedded (e.g. during re-indexing runs).

Key:   rag:emb:{sha256(model_name + chunk_text)[:32]}
Value: MessagePack-serialized float32 vector
TTL:   7 days (configurable via EMBED_CACHE_TTL)

Uses MessagePack (not JSON) for vectors — ~3x smaller and faster to
serialize/deserialize than JSON for float arrays.

Fallback: if msgpack is not installed, falls back to JSON automatically.
"""

import hashlib
import json
from typing import Any

from cache.redis_client import get_redis
from config import CACHE, EMBEDDING
from utils.logger import get_logger

logger = get_logger("cache.embedding")

# Try msgpack for efficient binary vector storage; fall back to JSON
try:
    import msgpack
    _USE_MSGPACK = True
except ImportError:
    _USE_MSGPACK = False
    logger.info("msgpack not installed — embedding cache using JSON. Install: pip install msgpack")


def _make_key(text: str) -> str:
    """Hash of (model_name + chunk_text) → stable cache key."""
    raw = f"{EMBEDDING.model_name}:{text}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{CACHE.embed_prefix}{digest}"


def _serialize(vector: list[float]) -> bytes:
    if _USE_MSGPACK:
        return msgpack.packb(vector, use_bin_type=True)
    return json.dumps(vector).encode()


def _deserialize(data: bytes) -> list[float]:
    if _USE_MSGPACK:
        return msgpack.unpackb(data, raw=False)
    return json.loads(data)


# ── Public API ────────────────────────────────────────────────

def get_many(texts: list[str]) -> dict[str, list[float]]:
    """
    Batch-fetch embeddings for a list of texts.
    Returns a dict of {text: vector} for cache hits only.
    Texts not in cache are absent from the returned dict.
    """
    if not CACHE.enabled:
        return {}
    r = get_redis()
    if r is None:
        return {}

    keys = [_make_key(t) for t in texts]
    # Redis MGET fetches all keys in one round-trip
    raw_values = r.mget(keys)

    result = {}
    hits = 0
    for text, raw in zip(texts, raw_values):
        if raw is not None:
            result[text] = _deserialize(raw)
            hits += 1

    if texts:
        logger.debug(f"L3 embedding cache: {hits}/{len(texts)} hits")

    return result


def set_many(text_vector_pairs: list[tuple[str, list[float]]]) -> None:
    """
    Batch-store (text, vector) pairs in Redis using a pipeline
    (single round-trip for all writes).
    """
    if not CACHE.enabled:
        return
    r = get_redis()
    if r is None:
        return

    pipe = r.pipeline(transaction=False)
    for text, vector in text_vector_pairs:
        key = _make_key(text)
        pipe.setex(key, CACHE.embed_ttl, _serialize(vector))
    pipe.execute()

    logger.debug(f"L3 SET: {len(text_vector_pairs)} embeddings cached (ttl={CACHE.embed_ttl}s)")


def embed_with_cache(
    texts: list[str],
    embed_fn,           # Callable[[list[str]], list[list[float]]]
) -> list[list[float]]:
    """
    Embed a list of texts, using the cache for any that are already stored.

    Flow:
      1. Check L3 cache for all texts (single MGET)
      2. Embed only the texts that missed
      3. Store new embeddings back to cache (single pipeline)
      4. Return full list in original order

    Args:
        texts:    List of strings to embed
        embed_fn: The actual embedding function (e.g. utils.embedder.embed_texts)

    Returns:
        List of float vectors, same length and order as `texts`
    """
    if not texts:
        return []

    # Step 1: Cache lookup
    cached = get_many(texts)

    # Step 2: Find misses
    miss_texts = [t for t in texts if t not in cached]
    miss_indices = [i for i, t in enumerate(texts) if t not in cached]

    # Step 3: Embed misses
    if miss_texts:
        logger.debug(f"L3 embedding {len(miss_texts)} cache misses...")
        new_vectors = embed_fn(miss_texts)
        # Store new embeddings in cache
        set_many(list(zip(miss_texts, new_vectors)))
        # Merge into cached dict
        for text, vector in zip(miss_texts, new_vectors):
            cached[text] = vector

    # Step 4: Reconstruct in original order
    return [cached[t] for t in texts]