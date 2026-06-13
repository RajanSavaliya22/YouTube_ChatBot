"""
BM25 Store
===========
Wraps BM25Index with:
  - Disk persistence (pickle)
  - Payload cache: chunk_id → ChunkPayload for fast retrieval
  - Video-level chunk tracking for bulk deletion
  - Optional metadata filtering (video_id, channel) applied post-scoring
"""

import pickle
import logging
from pathlib import Path

from schema import ChunkPayload
from vector_store.sparse.bm25_index import BM25Index
from config import BM25 as BM25_CFG
from utils.logger import get_logger

logger = get_logger("sparse.bm25_store")


class BM25Store:
    """
    Persistent BM25 store combining the scoring index with payload storage.

    Usage:
        store = BM25Store()
        store.load()                          # Load from disk (no-op if new)
        store.add_chunks(chunks, chunk_ids)   # Index new chunks
        results = store.search("my query")    # Search
        store.save()                          # Persist to disk
    """

    def __init__(self):
        self.index = BM25Index()
        # chunk_id → ChunkPayload (for returning full metadata on hits)
        self._payload_map: dict[str, ChunkPayload] = {}
        # video_id → [chunk_ids] (for fast bulk deletion)
        self._video_chunks: dict[str, list[str]] = {}

        self._index_dir = Path(BM25_CFG.index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _index_path(self) -> Path:
        return self._index_dir / BM25_CFG.index_file

    # ─── Indexing ─────────────────────────────────────────────────

    def add_chunks(self, chunks: list[ChunkPayload], chunk_ids: list[str]) -> None:
        """
        Add chunks to both the BM25 index and payload store.

        Args:
            chunks:    ChunkPayload list from Stage 3
            chunk_ids: Deterministic UUIDs matching Qdrant point IDs
        """
        assert len(chunks) == len(chunk_ids)

        texts = [c.chunk_text for c in chunks]
        self.index.add_documents(chunk_ids, texts)

        for chunk_id, chunk in zip(chunk_ids, chunks):
            self._payload_map[chunk_id] = chunk
            self._video_chunks.setdefault(chunk.video_id, []).append(chunk_id)

        logger.info(f"BM25 store: added {len(chunks)} chunks for video '{chunks[0].video_id}'.")

    def remove_video(self, video_id: str) -> None:
        """Remove all chunks for a video (called before re-indexing)."""
        chunk_ids = self._video_chunks.pop(video_id, [])
        if not chunk_ids:
            logger.warning(f"No BM25 chunks found for video_id='{video_id}'.")
            return

        self.index.remove_documents(chunk_ids)
        for chunk_id in chunk_ids:
            self._payload_map.pop(chunk_id, None)

        logger.info(f"BM25 store: removed {len(chunk_ids)} chunks for video '{video_id}'.")

    # ─── Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 50,
        filter_video_id: str | None = None,
        filter_channel: str | None = None,
    ) -> list[tuple[ChunkPayload, float, str]]:
        """
        BM25 search with optional metadata filtering.

        BM25 doesn't support native payload filtering, so we over-fetch
        (top_k × 3) and filter in Python before trimming to top_k.

        Args:
            query:            Raw user query
            top_k:            Max results to return after filtering
            filter_video_id:  Restrict to a specific video
            filter_channel:   Restrict to a specific channel

        Returns:
            List of (ChunkPayload, bm25_score, chunk_id) sorted by score desc
        """
        # Over-fetch to account for post-filter drop
        raw = self.index.score(query, top_k=top_k * 3)

        results = []
        for chunk_id, score in raw:
            payload = self._payload_map.get(chunk_id)
            if payload is None:
                continue

            # Apply filters
            if filter_video_id and payload.video_id != filter_video_id:
                continue
            if filter_channel and payload.channel != filter_channel:
                continue

            results.append((payload, score, chunk_id))
            if len(results) >= top_k:
                break

        return results

    # ─── Persistence ──────────────────────────────────────────────

    def save(self) -> None:
        """Serialize the full store to disk using pickle."""
        data = {
            "index": self.index,
            "payload_map": self._payload_map,
            "video_chunks": self._video_chunks,
        }
        with open(self._index_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 store saved → {self._index_path} ({self.index.num_docs} docs)")

    def load(self, qdrant_client=None) -> bool:
        """
        Load store from disk.
        If no pickle found AND qdrant_client is provided, rebuilds from Qdrant.

        Args:
            qdrant_client: Optional connected QdrantClient for auto-rebuild.
                           Pass this on Render/cloud where filesystem is ephemeral.

        Returns:
            True if loaded or rebuilt, False if empty start.
        """
        if self._index_path.exists():
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self.index = data["index"]
            self._payload_map = data["payload_map"]
            self._video_chunks = data["video_chunks"]
            logger.info(
                f"BM25 store loaded from {self._index_path} "
                f"({self.index.num_docs} docs, vocab={len(self.index.inverted_index)})"
            )
            return True

        # No pickle on disk
        if qdrant_client is not None:
            logger.info("No BM25 pickle found — rebuilding from Qdrant...")
            from vector_store.sparse.bm25_rebuild import rebuild_from_qdrant
            count = rebuild_from_qdrant(self, qdrant_client)
            return count > 0

        logger.info("No BM25 index on disk — starting fresh.")
        return False

    # ─── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            **self.index.stats(),
            "indexed_videos": len(self._video_chunks),
            "payload_entries": len(self._payload_map),
        }
    