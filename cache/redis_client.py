"""
Redis connection manager.
Falls back gracefully if redis package not installed or Redis is unreachable.
"""

from config import CACHE
from utils.logger import get_logger

logger = get_logger("cache.redis")

try:
    import redis as _redis_module
    _REDIS_PKG = True
except ImportError:
    _redis_module = None
    _REDIS_PKG = False

_client = None
_available: bool | None = None


def get_redis():
    global _client, _available

    if not _REDIS_PKG:
        return None
    if _available is False:
        return None

    if _client is None:
        _client = _redis_module.Redis(
            host=CACHE.redis_host,
            port=CACHE.redis_port,
            db=CACHE.redis_db,
            password=CACHE.redis_password,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    if _available is None:
        try:
            _client.ping()
            _available = True
            logger.info(f"Redis connected → {CACHE.redis_host}:{CACHE.redis_port}")
        except Exception as e:
            _available = False
            _client = None
            logger.warning(f"Redis unavailable — caching disabled. ({e})")

    return _client if _available else None


def is_available() -> bool:
    get_redis()
    return _available is True


def flush_all_rag_keys() -> int:
    r = get_redis()
    if r is None:
        return 0
    prefixes = [CACHE.exact_prefix, CACHE.semantic_prefix, CACHE.embed_prefix]
    deleted = 0
    for prefix in prefixes:
        keys = r.keys(f"{prefix}*")
        if keys:
            deleted += r.delete(*keys)
    logger.info(f"Flushed {deleted} RAG cache keys.")
    return deleted