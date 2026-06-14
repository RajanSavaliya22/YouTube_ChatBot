"""
Stage 2: Transcript Cleaning
==============================
Cleans raw transcript text before chunking:
  - Remove filler words, repeated phrases, YouTube artifacts
  - Fix punctuation (Whisper output often has none)
  - Merge short/fragmented caption lines into full sentences
  - Normalize whitespace and encoding issues

Input:  Transcript object (from Stage 1)
Output: Cleaned Transcript object saved to storage/cleaned/{video_id}.json
"""

import json
import re
from pathlib import Path

from config import TRANSCRIPT
from schema import Transcript, TranscriptSegment
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("stage2.cleaner")


# ─────────────────────────────────────────────
# Filler / noise patterns
# ─────────────────────────────────────────────

# YouTube auto-caption artifacts
CAPTION_ARTIFACTS = re.compile(
    r"\[(?:Music|Applause|Laughter|Sound|Noise|Intro|Outro|Background Music|"
    r"music|applause|laughter|sound|noise|intro|outro)\]",
    re.IGNORECASE,
)

# Filler words at word boundaries
FILLER_WORDS = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|hm+|mhm|uh-huh|uh-oh|uhh|umm|erm)\b",
    re.IGNORECASE,
)

# Repeated phrases: "you know you know", "like like"
REPEATED_PHRASES = re.compile(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", re.IGNORECASE)

# Excessive ellipsis / dashes from auto-captions
EXCESSIVE_PUNCTUATION = re.compile(r"([.!?,;])\1+")

# Leading/trailing punctuation artifacts
LEADING_PUNCT = re.compile(r"^[\s,;.!?-]+")

# Multiple spaces
MULTI_SPACE = re.compile(r" {2,}")

# Common YouTube transcript-specific noise
YOUTUBE_NOISE = re.compile(
    r"(?:subscribe|click the bell|hit the like button|comment below|"
    r"in this video|let me know|drop a|smash the|don't forget to)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# Punctuation restoration (Whisper output)
# ─────────────────────────────────────────────

_punct_model = None


def _get_punct_model():
    """Load deepmultilingualpunctuation model once."""
    global _punct_model
    if _punct_model is None:
        try:
            from deepmultilingualpunctuation import PunctuationModel
            logger.info("Loading punctuation restoration model...")
            _punct_model = PunctuationModel()
            logger.info("Punctuation model loaded.")
        except ImportError:
            logger.warning(
                "deepmultilingualpunctuation not installed. "
                "Skipping punctuation restoration. "
                "Install: pip install deepmultilingualpunctuation"
            )
            _punct_model = None
    return _punct_model


def restore_punctuation(text: str) -> str:
    """
    Add missing punctuation to Whisper output.
    Falls back to raw text if model unavailable.
    """
    model = _get_punct_model()
    if model is None:
        return text

    # Model works best on ≤500 word chunks
    words = text.split()
    chunk_size = 400
    restored_parts = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        restored = model.restore_punctuation(chunk)
        restored_parts.append(restored)

    return " ".join(restored_parts)


# ─────────────────────────────────────────────
# Segment-level cleaning
# ─────────────────────────────────────────────

def clean_segment_text(text: str) -> str:
    """Apply all cleaning rules to a single segment's text."""
    # Remove [Music], [Applause] etc.
    text = CAPTION_ARTIFACTS.sub("", text)

    # Remove filler words
    text = FILLER_WORDS.sub("", text)

    # Remove repeated phrases
    text = REPEATED_PHRASES.sub(r"\1", text)

    # Fix repeated punctuation
    text = EXCESSIVE_PUNCTUATION.sub(r"\1", text)

    # Remove leading punctuation artifacts
    text = LEADING_PUNCT.sub("", text)

    # Normalize whitespace
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def merge_short_segments(
    segments: list[TranscriptSegment],
    min_words: int = 5,
    max_gap_seconds: float = 1.5,
) -> list[TranscriptSegment]:
    """
    Merge segments that are too short into the next segment,
    as long as the gap between them is small enough.

    This fixes the fragmentation problem in YouTube auto-captions
    where one sentence is split across many 2-3 word segments.
    """
    if not segments:
        return segments

    merged = []
    buffer = segments[0]

    for seg in segments[1:]:
        gap = seg.start - buffer.end
        buffer_words = len(buffer.text.split())

        if buffer_words < min_words and gap <= max_gap_seconds:
            # Merge: extend buffer's text and end time
            buffer = TranscriptSegment(
                start=buffer.start,
                end=seg.end,
                text=buffer.text + " " + seg.text,
                speaker=buffer.speaker,
            )
        else:
            merged.append(buffer)
            buffer = seg

    merged.append(buffer)
    return merged


def deduplicate_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """
    Remove consecutive duplicate or near-duplicate segments.
    Auto-captions often repeat lines as they scroll.
    """
    if not segments:
        return segments

    deduped = [segments[0]]
    for seg in segments[1:]:
        prev_text = deduped[-1].text.lower().strip()
        curr_text = seg.text.lower().strip()

        # Exact duplicate
        if curr_text == prev_text:
            continue

        # One is a subset of the other (caption scroll overlap)
        if curr_text in prev_text or prev_text in curr_text:
            continue

        deduped.append(seg)

    removed = len(segments) - len(deduped)
    if removed:
        logger.info(f"Deduplication removed {removed} duplicate segments.")

    return deduped


# ─────────────────────────────────────────────
# Main cleaning pipeline
# ─────────────────────────────────────────────

@timed("stage2.cleaner")
def clean_transcript(
    transcript: Transcript,
    restore_punct: bool = True,
) -> Transcript:
    """
    Full cleaning pipeline applied to a Transcript.

    Steps:
      1. Clean each segment individually (noise removal, filler words)
      2. Deduplicate scrolling caption repeats
      3. Merge short fragmented segments
      4. Optionally restore punctuation (for Whisper output)
      5. Drop empty segments

    Args:
        transcript: Raw Transcript from Stage 1
        restore_punct: Whether to run deepmultilingualpunctuation

    Returns:
        Cleaned Transcript object
    """
    Path(TRANSCRIPT.cleaned_dir).mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_path = Path(TRANSCRIPT.cleaned_dir) / f"{transcript.video_id}.json"
    if cache_path.exists():
        logger.info(f"Loading cached cleaned transcript for {transcript.video_id}")
        return Transcript.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

    logger.info(f"Cleaning transcript: '{transcript.video_title}' ({len(transcript.segments)} segments)")

    # Step 1: Clean each segment
    cleaned_segs = []
    for seg in transcript.segments:
        text = clean_segment_text(seg.text)
        if text:
            cleaned_segs.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=text,
                speaker=seg.speaker,
            ))

    logger.info(f"After cleaning: {len(cleaned_segs)} segments (was {len(transcript.segments)})")

    # Step 2: Deduplicate
    cleaned_segs = deduplicate_segments(cleaned_segs)

    # Step 3: Merge short segments
    cleaned_segs = merge_short_segments(cleaned_segs, min_words=5, max_gap_seconds=1.5)
    logger.info(f"After merging short segments: {len(cleaned_segs)} segments")

    # Step 4: Restore punctuation on the full joined text, then redistribute
    if restore_punct:
        full_text = " ".join(s.text for s in cleaned_segs)
        restored = restore_punctuation(full_text)
        # Re-split by approximately same count (punctuation model changes word count slightly)
        # Simple approach: re-distribute restored text proportionally
        words_restored = restored.split()
        total_orig_words = sum(len(s.text.split()) for s in cleaned_segs)
        pointer = 0
        for seg in cleaned_segs:
            orig_count = len(seg.text.split())
            proportion = orig_count / max(total_orig_words, 1)
            take = max(1, round(proportion * len(words_restored)))
            seg.text = " ".join(words_restored[pointer:pointer + take])
            pointer += take

    # Step 5: Final empty check
    final_segs = [s for s in cleaned_segs if s.text.strip()]

    cleaned_transcript = Transcript(
        video_id=transcript.video_id,
        video_title=transcript.video_title,
        channel=transcript.channel,
        video_url=transcript.video_url,
        published_date=transcript.published_date,
        language=transcript.language,
        segments=final_segs,
    )

    # Save
    cache_path.write_text(
        json.dumps(cleaned_transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Cleaned transcript saved: {cache_path} ({len(final_segs)} segments)")

    return cleaned_transcript


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pipeline.transcript import get_transcript

    if len(sys.argv) < 2:
        print("Usage: python pipeline/cleaner.py <youtube_url>")
        sys.exit(1)

    t = get_transcript(sys.argv[1])
    cleaned = clean_transcript(t)
    print(f"\n✓ Cleaned transcript: {len(cleaned.segments)} segments")
    print(f"  Preview: {cleaned.full_text[:300]}...")
