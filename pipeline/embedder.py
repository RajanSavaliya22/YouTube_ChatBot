"""
Stage 4: Embeddings
====================
Converts child chunks into dense vector embeddings for semantic search.

  - Uses BAAI/bge-large-en-v1.5 (1024-dim, best open-source English model)
  - Batched processing to fit in RAM/VRAM
  - L2-normalized vectors (required for cosine similarity in Qdrant)
  - Document-side encoding (no query prefix here — that's in the retriever)
  - Results cached to avoid re-embedding on re-runs

Input:  List[ChunkPayload] (from Stage 3)
Output: List[List[float]] — one vector per chunk, same order as input
"""

import json
import hashlib
from pathlib import Path

from config import TRANSCRIPT, EMBEDDING
from schema import ChunkPayload
from utils.logger import get_logger
from utils.timer import timed
from utils.embedder import embed_texts

logger = get_logger("stage4.embedder")

EMBED_CACHE_DIR = Path("storage/embeddings")


def _chunk_cache_key(chunks: list[ChunkPayload]) -> str:
    """Hash the chunk texts + model name to produce a stable cache key."""
    combined = EMBEDDING.model_name + "".join(c.chunk_text for c in chunks)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _load_cached(cache_path: Path) -> list[list[float]] | None:
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return None


def _save_cached(cache_path: Path, embeddings: list[list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(embeddings), encoding="utf-8")
    logger.info(f"Embeddings cached to {cache_path}")


@timed("stage4.embedder")
def embed_chunks(
    chunks: list[ChunkPayload],
    use_cache: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for a list of ChunkPayload objects.

    Only the child chunk text is embedded (not the parent).
    The parent text is only used at generation time.

    Args:
        chunks:     List of ChunkPayload from Stage 3
        use_cache:  Skip recomputing if embeddings already cached

    Returns:
        List of float vectors, same length and order as `chunks`
    """
    if not chunks:
        logger.warning("No chunks to embed.")
        return []

    # Cache lookup
    if use_cache:
        video_id = chunks[0].video_id
        cache_key = _chunk_cache_key(chunks)
        cache_path = EMBED_CACHE_DIR / f"{video_id}_{cache_key}.json"
        cached = _load_cached(cache_path)
        if cached is not None:
            assert len(cached) == len(chunks), "Cache length mismatch — invalidating."
            return cached

    logger.info(
        f"Embedding {len(chunks)} chunks with {EMBEDDING.model_name} "
        f"(device={EMBEDDING.device}, batch={EMBEDDING.batch_size})"
    )

    texts = [c.chunk_text for c in chunks]
    embeddings = embed_texts(texts)

    assert len(embeddings) == len(chunks), "Embedding count mismatch!"
    logger.info(f"Produced {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # Cache result
    if use_cache:
        _save_cached(cache_path, embeddings)

    return embeddings


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pipeline.stage01_transcript import get_transcript
    from pipeline.stage02_cleaner import clean_transcript
    from pipeline.stage03_chunker import chunk_transcript

    if len(sys.argv) < 2:
        print("Usage: python pipeline/embedder.py <youtube_url>")
        sys.exit(1)

    t = get_transcript(sys.argv[1])
    cleaned = clean_transcript(t)
    chunks = chunk_transcript(cleaned)
    embeddings = embed_chunks(chunks)

    print(f"\n✓ {len(embeddings)} embeddings produced")
    print(f"  Dimensions: {len(embeddings[0])}")
    print(f"  First vector (first 5 dims): {embeddings[0][:5]}")
