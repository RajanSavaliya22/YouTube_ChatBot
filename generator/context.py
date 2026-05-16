"""
Context Builder
================
Assembles the top re-ranked chunks into a structured context block
for injection into the LLM prompt.

Responsibilities:
  - Select top N chunks from re-ranked results
  - Use parent_text (full context) not chunk_text (small embedded unit)
  - Truncate each chunk to max_chunk_tokens to stay within LLM context window
  - Label each source with [Source N] tags for citation
  - Build timestamp citation map: Source N → clickable YouTube URL + timestamp
  - Detect low-confidence situations (all scores below threshold)

Output structure injected into the system prompt:
  [Source 1] Video: "Title" | Channel: Name | Time: 04:12 → 05:40
  <text of parent chunk 1>

  [Source 2] Video: "Title" | Channel: Name | Time: 08:33 → 09:10
  <text of parent chunk 2>
  ...
"""

from dataclasses import dataclass
from config import GENERATOR
from reranker.model import RankedChunk
from utils.logger import get_logger

logger = get_logger("generator.context")


@dataclass
class SourceCitation:
    """Metadata for one cited source, included in the response."""
    index: int           # [Source N] label used in LLM output
    video_title: str
    channel: str
    timestamp_label: str
    url_with_timestamp: str
    rerank_score: float


@dataclass
class BuiltContext:
    """Fully assembled context ready to inject into the LLM prompt."""
    context_text: str              # Formatted [Source N] blocks
    citations: list[SourceCitation]  # Metadata for response footer
    is_confident: bool             # False = scores too low, should refuse
    chunk_count: int


def build_context(
    ranked_chunks: list[RankedChunk],
    max_chunks: int | None = None,
) -> BuiltContext:
    """
    Assemble ranked chunks into a formatted LLM context block.

    Args:
        ranked_chunks: Output from Stage 8 reranker (sorted by score desc)
        max_chunks:    Max chunks to include (defaults to GENERATOR.max_context_chunks)

    Returns:
        BuiltContext with formatted text, citations, and confidence flag
    """
    max_chunks = max_chunks or GENERATOR.max_context_chunks

    if not ranked_chunks:
        return BuiltContext(
            context_text="No relevant content found in the indexed videos.",
            citations=[],
            is_confident=False,
            chunk_count=0,
        )

    # Confidence check: if best score is below threshold, signal low confidence
    best_score = ranked_chunks[0].rerank_score
    is_confident = best_score >= GENERATOR.min_confidence_score

    if not is_confident:
        logger.info(
            f"Low confidence: best rerank score {best_score:.3f} "
            f"< threshold {GENERATOR.min_confidence_score}"
        )

    top_chunks = ranked_chunks[:max_chunks]
    context_parts = []
    citations = []

    for i, chunk in enumerate(top_chunks, start=1):
        p = chunk.payload

        # Truncate parent_text to max_chunk_tokens
        parent_text = _truncate(p.parent_text, GENERATOR.max_chunk_tokens)

        # Build source header
        ts_url = f"{p.video_url}?t={int(p.timestamp_start)}"
        header = (
            f"[Source {i}] "
            f"Video: \"{p.video_title}\" | "
            f"Channel: {p.channel} | "
            f"Time: {p.timestamp_label}"
        )

        context_parts.append(f"{header}\n{parent_text}")

        citations.append(SourceCitation(
            index=i,
            video_title=p.video_title,
            channel=p.channel,
            timestamp_label=p.timestamp_label,
            url_with_timestamp=ts_url,
            rerank_score=chunk.rerank_score,
        ))

    context_text = "\n\n".join(context_parts)

    logger.info(
        f"Context built: {len(top_chunks)} chunks, "
        f"confident={is_confident}, best_score={best_score:.3f}"
    )

    return BuiltContext(
        context_text=context_text,
        citations=citations,
        is_confident=is_confident,
        chunk_count=len(top_chunks),
    )


def _truncate(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    Uses word count as a fast proxy (avg ~1.3 tokens/word for English).
    """
    words = text.split()
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."