"""
Stage 5b: Indexer
==================
Upserts chunks into Qdrant (dense) and BM25 (sparse) simultaneously.
Uses deterministic UUIDs so re-indexing the same video is idempotent.
"""

import uuid
import logging
from typing import Generator

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from schema import ChunkPayload
from vector_store.sparse.bm25_store import BM25Store
from config import COLLECTION
from utils.logger import get_logger

logger = get_logger("vector_store.indexer")

BATCH_SIZE = 128


def _make_chunk_id(video_id: str, chunk_index: int) -> str:
    """
    Deterministic UUID from (video_id, chunk_index).
    Re-indexing the same chunk produces the same ID → idempotent upsert.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}:{chunk_index}"))


def upsert_chunks(
    client: QdrantClient,
    bm25_store: BM25Store,
    chunks: list[ChunkPayload],
    embeddings: list[list[float]],
    save_bm25: bool = True,
) -> None:
    """
    Upsert chunks into both stores:
      - Qdrant: vector + full payload
      - BM25:   text + payload reference

    Args:
        client:     Active QdrantClient
        bm25_store: Loaded BM25Store instance
        chunks:     ChunkPayload list from Stage 3
        embeddings: Float vectors from Stage 4 (same order as chunks)
        save_bm25:  Persist BM25 index to disk after upsert
    """
    assert len(chunks) == len(embeddings), (
        f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
    )

    total = len(chunks)
    chunk_ids = [_make_chunk_id(c.video_id, c.chunk_index) for c in chunks]

    # ── Qdrant upsert (batched) ────────────────────────────────────
    logger.info(f"Upserting {total} chunks into Qdrant (batch={BATCH_SIZE})...")
    batch_count = 0

    for b_chunks, b_vecs, b_ids in _batcher(chunks, embeddings, chunk_ids):
        points = [
            PointStruct(
                id=cid,
                vector=vec,
                payload=chunk.to_dict(),
            )
            for cid, vec, chunk in zip(b_ids, b_vecs, b_chunks)
        ]
        client.upsert(
            collection_name=COLLECTION.name,
            points=points,
            wait=True,   # Ensure indexing completes before next batch
        )
        batch_count += 1

    logger.info(f"Qdrant: {total} chunks upserted in {batch_count} batches.")

    # ── BM25 upsert ───────────────────────────────────────────────
    logger.info(f"Adding {total} chunks to BM25 store...")
    bm25_store.add_chunks(chunks, chunk_ids)

    if save_bm25:
        bm25_store.save()

    logger.info("Both stores updated successfully.")


def delete_video(
    client: QdrantClient,
    bm25_store: BM25Store,
    video_id: str,
) -> None:
    """
    Remove all chunks for a video from both Qdrant and BM25.
    Use before re-indexing an updated video.
    """
    # Qdrant: filter-delete by video_id
    client.delete(
        collection_name=COLLECTION.name,
        points_selector=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        wait=True,
    )
    logger.info(f"Qdrant: deleted chunks for video_id='{video_id}'")

    # BM25: remove from in-memory index
    bm25_store.remove_video(video_id)
    bm25_store.save()
    logger.info(f"BM25: deleted chunks for video_id='{video_id}'")


def video_is_indexed(client: QdrantClient, video_id: str) -> bool:
    """
    Check if a video already has chunks in Qdrant.
    Used to skip re-indexing unchanged videos.
    """
    result = client.scroll(
        collection_name=COLLECTION.name,
        scroll_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(result[0]) > 0


def _batcher(
    chunks: list,
    embeddings: list,
    chunk_ids: list,
) -> Generator[tuple[list, list, list], None, None]:
    """Yield fixed-size batches of (chunks, embeddings, ids)."""
    for i in range(0, len(chunks), BATCH_SIZE):
        yield (
            chunks[i:i + BATCH_SIZE],
            embeddings[i:i + BATCH_SIZE],
            chunk_ids[i:i + BATCH_SIZE],
        )
