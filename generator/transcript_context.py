"""
Transcript Context Builder
============================
Builds a broad context from sampled transcript chunks for overview queries.
Unlike the standard context builder (which uses top-ranked retrieval chunks),
this samples evenly across the ENTIRE video to give the LLM full coverage.

Sampling strategy:
  ≤ 60 chunks  → use all chunks (short video, fits in context)
  ≤ 150 chunks → every 2nd chunk
  > 150 chunks → every Nth chunk to target ~60 chunks total

Why ~60 chunks?
  60 chunks × ~300 tokens/chunk = ~18k tokens
  llama-3.3-70b context window = 128k tokens → safe headroom

Chunks are sorted by timestamp so the LLM sees the video in order.
"""

from dataclasses import dataclass
from schema import ChunkPayload
from utils.logger import get_logger

logger = get_logger("generator.transcript_context")

TARGET_CHUNKS = 30      # Max chunks to feed for overview queries
MAX_CHUNK_WORDS = 120   # Truncate each chunk (overview needs breadth not depth)


@dataclass
class TranscriptContext:
    """Full-video context for overview queries."""
    context_text: str
    total_chunks_available: int
    chunks_used: int
    video_title: str
    channel: str
    video_url: str


def build_transcript_context(
    bm25_store,
    video_id: str | None = None,
    channel: str | None = None,
) -> TranscriptContext | None:
    """
    Build a broad transcript context by sampling chunks from the BM25 payload map.

    Args:
        bm25_store:  Loaded BM25Store (has _payload_map and _video_chunks)
        video_id:    Filter to a specific video (required if multiple videos indexed)
        channel:     Filter to a specific channel (used if video_id not given)

    Returns:
        TranscriptContext with sampled chunks, or None if nothing found
    """
    # Collect all payloads matching the filter
    all_payloads: list[ChunkPayload] = []

    for chunk_id, payload in bm25_store._payload_map.items():
        if video_id and payload.video_id != video_id:
            continue
        if channel and payload.channel.lower() != channel.lower():
            continue
        all_payloads.append(payload)

    if not all_payloads:
        logger.warning(
            f"No chunks found for video_id={video_id} channel={channel}. "
            f"Total indexed: {len(bm25_store._payload_map)}"
        )
        return None

    # Sort by timestamp so the LLM reads the video in order
    all_payloads.sort(key=lambda p: p.timestamp_start)

    total = len(all_payloads)

    # Determine sampling step
    if total <= TARGET_CHUNKS:
        sampled = all_payloads
    else:
        step = max(1, total // TARGET_CHUNKS)
        sampled = all_payloads[::step]
        # Always include last chunk (video conclusion)
        if all_payloads[-1] not in sampled:
            sampled.append(all_payloads[-1])

    logger.info(
        f"Transcript context: {total} total chunks → "
        f"sampled {len(sampled)} (every {total // TARGET_CHUNKS if total > TARGET_CHUNKS else 1})"
    )

    # Build context text
    parts = []
    for i, p in enumerate(sampled, start=1):
        text = _truncate(p.chunk_text, MAX_CHUNK_WORDS)
        parts.append(f"[{p.timestamp_label}]\n{text}")

    context_text = "\n\n".join(parts)

    # Use metadata from first chunk
    first = sampled[0]

    return TranscriptContext(
        context_text=context_text,
        total_chunks_available=total,
        chunks_used=len(sampled),
        video_title=first.video_title,
        channel=first.channel,
        video_url=first.video_url,
    )


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."