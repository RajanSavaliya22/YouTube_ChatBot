"""
Stage 3: Chunking
==================
Splits cleaned transcripts into hierarchical chunk pairs:

  Parent chunk (~1200 tokens): Rich context sent to the LLM
  Child chunk  (~300 tokens):  Small, precise unit used for embedding & retrieval

Strategy:
  1. Split transcript into sentence-aligned child chunks with sliding overlap
  2. Each child chunk tracks its parent (wider context window around it)
  3. Every chunk carries timestamp metadata for deep-link citations

Input:  Cleaned Transcript (from Stage 2)
Output: List[ChunkPayload] saved to storage/chunks/{video_id}.json
"""

import json
from pathlib import Path

from config import TRANSCRIPT, CHUNKING, EMBEDDING
from schema import Transcript, TranscriptSegment, ChunkPayload, fmt_timestamp
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("stage3.chunker")


# ─────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import tiktoken
        # cl100k_base works well as a general-purpose tokenizer
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ─────────────────────────────────────────────
# Sentence-aware splitting
# ─────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Prefers spaCy if available; falls back to regex.
    """
    # try:
    #     import spacy
    #     # Load smallest English model — just for sentence boundaries
    #     try:
    #         nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    #     except OSError:
    #         # Model not downloaded — use sentencizer component only
    #         nlp = spacy.blank("en")
    #         nlp.add_pipe("sentencizer")
    #     doc = nlp(text)
    #     return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # except ImportError:
    #     pass

    # Regex fallback: split on sentence-ending punctuation
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────
# Segment → sentence bridge
# ─────────────────────────────────────────────

def _build_sentence_blocks(segments: list[TranscriptSegment]) -> list[dict]:
    """
    Convert transcript segments into sentence-level blocks,
    each carrying an approximate timestamp.

    Since one segment may contain multiple sentences, we assign
    the segment's timestamp to all sentences within it.
    """
    blocks = []
    for seg in segments:
        sentences = _split_into_sentences(seg.text)
        if not sentences:
            continue
        # Distribute time evenly across sentences in this segment
        duration = seg.end - seg.start
        per_sent = duration / len(sentences)
        for i, sent in enumerate(sentences):
            blocks.append({
                "text": sent,
                "start": seg.start + i * per_sent,
                "end": seg.start + (i + 1) * per_sent,
            })
    return blocks


# ─────────────────────────────────────────────
# Child chunk builder (sliding window)
# ─────────────────────────────────────────────

def _build_child_chunks(
    blocks: list[dict],
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """
    Greedy sliding-window chunker over sentence blocks.
    Fills each chunk to ~target_tokens, then advances with overlap.

    Returns list of child dicts: {text, start, end, block_start_idx, block_end_idx}
    """
    chunks = []
    i = 0
    n = len(blocks)

    while i < n:
        chunk_blocks = []
        token_count = 0

        j = i
        while j < n and token_count < target_tokens:
            block = blocks[j]
            block_tokens = count_tokens(block["text"])
            chunk_blocks.append(block)
            token_count += block_tokens
            j += 1

        if not chunk_blocks:
            break

        chunk_text = " ".join(b["text"] for b in chunk_blocks)
        chunks.append({
            "text": chunk_text,
            "start": chunk_blocks[0]["start"],
            "end": chunk_blocks[-1]["end"],
            "block_start_idx": i,
            "block_end_idx": j - 1,
        })

        # Advance pointer, stepping back by overlap
        overlap_blocks = 0
        overlap_count = 0
        for b in reversed(chunk_blocks):
            overlap_count += count_tokens(b["text"])
            overlap_blocks += 1
            if overlap_count >= overlap_tokens:
                break

        i = max(i + 1, j - overlap_blocks)

    return chunks


# ─────────────────────────────────────────────
# Parent chunk builder
# ─────────────────────────────────────────────

def _build_parent_for_child(
    child_idx: int,
    child_chunks: list[dict],
    blocks: list[dict],
    target_tokens: int,
) -> dict:
    """
    Construct a parent chunk centered around a child chunk.
    Expands outward from the child's block range until target_tokens is reached.
    """
    child = child_chunks[child_idx]
    start_block = child["block_start_idx"]
    end_block = child["block_end_idx"]

    left = start_block
    right = end_block
    token_count = count_tokens(child["text"])
    n = len(blocks)

    # Expand symmetrically outward
    while token_count < target_tokens:
        expanded = False
        if left > 0:
            left -= 1
            token_count += count_tokens(blocks[left]["text"])
            expanded = True
        if right < n - 1 and token_count < target_tokens:
            right += 1
            token_count += count_tokens(blocks[right]["text"])
            expanded = True
        if not expanded:
            break

    parent_text = " ".join(blocks[k]["text"] for k in range(left, right + 1))
    return {
        "text": parent_text,
        "start": blocks[left]["start"],
        "end": blocks[right]["end"],
    }


# ─────────────────────────────────────────────
# Main chunking entry point
# ─────────────────────────────────────────────

@timed("stage3.chunker")
def chunk_transcript(transcript: Transcript) -> list[ChunkPayload]:
    """
    Full chunking pipeline for a cleaned transcript.

    Steps:
      1. Convert segments → sentence blocks with timestamps
      2. Build child chunks (embedding units, ~300 tokens, sliding window)
      3. Build parent chunk for each child (LLM context, ~1200 tokens)
      4. Wrap into ChunkPayload objects with full metadata

    Args:
        transcript: Cleaned Transcript from Stage 2

    Returns:
        List of ChunkPayload objects ready for embedding and indexing
    """
    Path(TRANSCRIPT.chunks_dir).mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_path = Path(TRANSCRIPT.chunks_dir) / f"{transcript.video_id}.json"
    if cache_path.exists():
        logger.info(f"Loading cached chunks for {transcript.video_id}")
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [ChunkPayload.from_dict(d) for d in data]

    logger.info(
        f"Chunking: '{transcript.video_title}' | "
        f"{len(transcript.segments)} segments | "
        f"child={CHUNKING.child_chunk_size}tok parent={CHUNKING.parent_chunk_size}tok"
    )

    # Step 1: Sentence blocks
    blocks = _build_sentence_blocks(transcript.segments)
    logger.info(f"Built {len(blocks)} sentence blocks.")

    if not blocks:
        logger.warning("No sentence blocks produced — empty transcript?")
        return []

    # Step 2: Child chunks
    child_chunks = _build_child_chunks(
        blocks,
        target_tokens=CHUNKING.child_chunk_size,
        overlap_tokens=CHUNKING.overlap_tokens,
    )
    logger.info(f"Built {len(child_chunks)} child chunks.")

    # Step 3 + 4: Parent + ChunkPayload assembly
    payloads: list[ChunkPayload] = []
    for idx, child in enumerate(child_chunks):
        parent = _build_parent_for_child(
            idx, child_chunks, blocks, CHUNKING.parent_chunk_size
        )

        ts_label = f"{fmt_timestamp(child['start'])} → {fmt_timestamp(child['end'])}"

        payloads.append(ChunkPayload(
            video_id=transcript.video_id,
            video_title=transcript.video_title,
            channel=transcript.channel,
            video_url=transcript.video_url,
            published_date=transcript.published_date,
            language=transcript.language,
            chunk_text=child["text"],
            parent_text=parent["text"],
            chunk_index=idx,
            timestamp_start=child["start"],
            timestamp_end=child["end"],
            timestamp_label=ts_label,
            embedding_model=EMBEDDING.model_name,
        ))

    # Save
    cache_path.write_text(
        json.dumps([p.to_dict() for p in payloads], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Saved {len(payloads)} chunks → {cache_path}")

    return payloads


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pipeline.stage01_transcript import get_transcript
    from pipeline.stage02_cleaner import clean_transcript

    if len(sys.argv) < 2:
        print("Usage: python pipeline/chunker.py <youtube_url>")
        sys.exit(1)

    t = get_transcript(sys.argv[1])
    cleaned = clean_transcript(t)
    chunks = chunk_transcript(cleaned)

    print(f"\n✓ {len(chunks)} chunks created")
    if chunks:
        c = chunks[0]
        print(f"\nChunk #0 ({c.timestamp_label}):")
        print(f"  Child  ({count_tokens(c.chunk_text)} tok): {c.chunk_text[:150]}...")
        print(f"  Parent ({count_tokens(c.parent_text)} tok): {c.parent_text[:150]}...")
