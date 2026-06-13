"""
BM25 Rebuild from Qdrant
=========================
On Render free tier, the filesystem is ephemeral — the BM25 pickle
is lost on every restart. This module rebuilds the BM25 index by
scrolling all payloads from Qdrant on startup.

Called automatically by BM25Store.load() if the pickle is not found.

Performance: ~2-5 seconds for 1000 chunks (Qdrant scroll is fast).
"""

from utils.logger import get_logger

logger = get_logger("sparse.bm25_rebuild")


def rebuild_from_qdrant(bm25_store, qdrant_client) -> int:
    """
    Rebuild BM25 index by scrolling all payloads from Qdrant.

    Args:
        bm25_store:     Empty BM25Store instance to populate
        qdrant_client:  Connected QdrantClient

    Returns:
        Number of chunks indexed
    """
    from config import COLLECTION
    from schema import ChunkPayload

    logger.info("Rebuilding BM25 index from Qdrant payloads...")

    all_chunks = []
    all_ids = []
    offset = None
    batch_size = 500

    while True:
        result, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION.name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not result:
            break

        for point in result:
            try:
                payload = ChunkPayload(**point.payload)
                all_chunks.append(payload)
                all_ids.append(str(point.id))
            except Exception as e:
                logger.warning(f"Skipping point {point.id}: {e}")

        logger.info(f"  Scrolled {len(all_chunks)} chunks so far...")

        if next_offset is None:
            break
        offset = next_offset

    if not all_chunks:
        logger.warning("No chunks found in Qdrant — BM25 index is empty.")
        return 0

    # Batch add to BM25
    bm25_store.add_chunks(all_chunks, all_ids)

    # Save to disk (will be used until next restart)
    bm25_store.save()

    logger.info(
        f"BM25 rebuilt: {len(all_chunks)} chunks, "
        f"{len(bm25_store._video_chunks)} videos, "
        f"vocab={len(bm25_store.index.inverted_index)}"
    )
    return len(all_chunks)