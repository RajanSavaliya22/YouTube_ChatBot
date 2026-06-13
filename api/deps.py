"""
API Dependencies
=================
Shared singletons injected into all FastAPI routes.
All heavy objects are created once at startup and reused across requests.
"""

from vector_store.client import get_client as _get_qdrant_client, create_collection
from vector_store.sparse.bm25_store import BM25Store
from cache.manager import CacheManager
from query_optimizer.pipeline import QueryOptimizer
from generator.pipeline import Generator
from config import API
from utils.logger import get_logger
from fastapi import Header, HTTPException, status

logger = get_logger("api.deps")

# ── Singletons ────────────────────────────────────────────────

_qdrant = None
_bm25: BM25Store | None = None
_cache: CacheManager | None = None
_optimizer: QueryOptimizer | None = None
_generator: Generator | None = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = _get_qdrant_client()
        create_collection(_qdrant)
    return _qdrant


def get_bm25() -> BM25Store:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Store()
        _bm25.load(qdrant_client=get_qdrant())
    return _bm25


def get_cache_manager() -> CacheManager:
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def get_query_optimizer() -> QueryOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def startup() -> None:
    """Pre-load all singletons at startup so first request is fast."""
    logger.info("API startup — initialising singletons...")
    get_qdrant()
    logger.info("✓ Qdrant connected")
    get_bm25()
    logger.info("✓ BM25 store ready")
    get_cache_manager()
    logger.info("✓ Cache manager ready")
    logger.info("Startup complete.")


def shutdown() -> None:
    if _bm25 is not None:
        _bm25.save()
        logger.info("BM25 store saved on shutdown.")


# ── Optional API key auth ─────────────────────────────────────

async def verify_api_key(x_api_key: str | None = Header(default=None)):
    """
    Optional API key auth. Enable by setting API_KEY in .env.
    All requests pass through when API_KEY is not set.
    """
    if API.api_key is None:
        return
    if x_api_key != API.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-Api-Key header.",
        )