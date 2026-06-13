"""
Embedding Utility
==================
Supports two backends, switched via EMBEDDER_BACKEND env var:

  local   (default) — sentence-transformers loaded in-process
  voyage  — Voyage AI embedding API (no local model, zero RAM)

Use 'voyage' backend for Render free tier deployment.
Use 'local' for local development.

Voyage model: voyage-4-lite
  - Default output: 1024-dim (matches bge-large collection)
  - Matryoshka: also supports 2048, 512, 256
  - 32K context window
  - Free tier: 200M tokens
  - ~50ms per batch API call
"""

import os
import math
from utils.logger import get_logger
from config import EMBEDDING

logger = get_logger("embedder")

BACKEND = os.getenv("EMBEDDER_BACKEND", "local")  # "local" | "voyage"

# ── Local backend ─────────────────────────────────────────────

_local_model = None


def get_model():
    """Load local sentence-transformers model once."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING.model_name} on {EMBEDDING.device}")
        _local_model = SentenceTransformer(EMBEDDING.model_name, device=EMBEDDING.device)
        logger.info("Embedding model loaded.")
    return _local_model


def _embed_local(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = get_model()
    if is_query:
        texts = [f"{EMBEDDING.query_prefix}{t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return embeddings.tolist()


# ── Voyage backend ────────────────────────────────────────────

VOYAGE_MODEL       = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-lite")
VOYAGE_DIMENSIONS  = int(os.getenv("VECTOR_SIZE", "1024"))   # 2048|1024|512|256
VOYAGE_BATCH_SIZE  = int(os.getenv("VOYAGE_EMBED_BATCH", "128"))  # Voyage max per request

_voyage_client = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set in .env")
        _voyage_client = voyageai.Client(api_key=api_key)
        logger.info(
            f"Voyage embedding client ready "
            f"(model={VOYAGE_MODEL}, dim={VOYAGE_DIMENSIONS})"
        )
    return _voyage_client


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalize vectors — required for cosine similarity in Qdrant."""
    result = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        result.append([x / norm for x in vec] if norm > 0 else vec)
    return result


def _embed_voyage(texts: list[str], input_type: str) -> list[list[float]]:
    """
    Embed texts via Voyage AI API.
    input_type: "document" for chunks, "query" for user queries.
    Automatically batches to respect Voyage's per-request limit.
    """
    client = _get_voyage_client()
    all_embeddings = []

    for i in range(0, len(texts), VOYAGE_BATCH_SIZE):
        batch = texts[i:i + VOYAGE_BATCH_SIZE]
        batch_num = i // VOYAGE_BATCH_SIZE + 1
        total_batches = math.ceil(len(texts) / VOYAGE_BATCH_SIZE)

        logger.info(
            f"Voyage embed batch {batch_num}/{total_batches} "
            f"({len(batch)} texts, input_type={input_type})"
        )

        result = client.embed(
            batch,
            model=VOYAGE_MODEL,
            input_type=input_type,
            output_dimension=VOYAGE_DIMENSIONS,   # Matryoshka truncation
        )
        all_embeddings.extend(result.embeddings)

    # Voyage returns normalized vectors by default, but normalize again to be safe
    return _normalize(all_embeddings)


# ── Public API ────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks for indexing."""
    logger.info(f"Embedding {len(texts)} texts [{BACKEND}]...")
    if BACKEND == "voyage":
        return _embed_voyage(texts, input_type="document")
    return _embed_local(texts, is_query=False)


def embed_query(query: str) -> list[float]:
    """Embed a single user query for retrieval."""
    if BACKEND == "voyage":
        return _embed_voyage([query], input_type="query")[0]
    return _embed_local([query], is_query=True)[0]