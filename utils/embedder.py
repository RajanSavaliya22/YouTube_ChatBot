"""
Embedding utility used by both the pipeline (indexing) and retriever (querying).
Wraps sentence-transformers with batching, normalization, and model caching.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from utils.logger import get_logger
from config import EMBEDDING

logger = get_logger("embedder")

_model = None  # Module-level singleton


def get_model():
    """Load embedding model once and reuse."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING.model_name} on {EMBEDDING.device}")
        _model = SentenceTransformer(EMBEDDING.model_name, device=EMBEDDING.device, cache_folder=EMBEDDING.cache_folder)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts (chunks).
    L2-normalized — required for cosine similarity in Qdrant.
    """
    model = get_model()
    logger.info(f"Embedding {len(texts)} texts (batch_size={EMBEDDING.batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query.
    BGE models require a special prefix on the query side for best results.
    """
    model = get_model()
    prefixed = f"{EMBEDDING.query_prefix}{query}"
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
    )
    return embedding[0].tolist()