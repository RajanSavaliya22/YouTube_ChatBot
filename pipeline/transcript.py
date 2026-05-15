"""
Stage 1: Transcript Generation
================================
Pulls transcripts from YouTube videos using two strategies:
  A) yt-dlp  — fetches native YouTube captions (fast, free, no GPU)
  B) Whisper  — transcribes audio when no captions exist (accurate, slower)

Output: Transcript object saved as JSON to storage/transcripts/{video_id}.json
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from config import TRANSCRIPT
from schema import Transcript, TranscriptSegment, fmt_timestamp
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("stage1.transcript")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any YouTube URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def _seconds_to_float(ts: str) -> float:
    """Convert HH:MM:SS.mmm or MM:SS.mmm to float seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _get_video_metadata(url: str) -> dict:
    """Fetch video title, channel, publish date via yt-dlp (no download)."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata fetch failed: {result.stderr}")
    info = json.loads(result.stdout)
    return {
        "video_id": info.get("id", ""),
        "video_title": info.get("title", ""),
        "channel": info.get("uploader", info.get("channel", "")),
        "video_url": info.get("webpage_url", url),
        "published_date": info.get("upload_date", ""),  # YYYYMMDD
        "language": info.get("language", "en") or "en",
    }


# ─────────────────────────────────────────────
# Strategy A: yt-dlp native captions
# ─────────────────────────────────────────────

def _parse_vtt(vtt_text: str) -> list[TranscriptSegment]:
    """Parse WebVTT caption file into TranscriptSegment list."""
    segments = []
    # Match cue blocks: timestamp --> timestamp \n text
    cue_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d+)\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d+)[^\n]*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    for match in cue_pattern.finditer(vtt_text):
        start_str, end_str, raw_text = match.groups()
        text = re.sub(r"<[^>]+>", "", raw_text).strip()  # Strip HTML tags
        text = re.sub(r"\n", " ", text).strip()
        if text:
            segments.append(TranscriptSegment(
                start=_seconds_to_float(start_str),
                end=_seconds_to_float(end_str),
                text=text,
            ))
    return segments


def fetch_native_captions(url: str, language: str = "en") -> list[TranscriptSegment] | None:
    """
    Download auto-generated or manual captions via yt-dlp.
    Returns None if no captions are available.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--write-subs",
            "--sub-lang", language,
            "--sub-format", "vtt",
            "--skip-download",
            "--no-playlist",
            "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Find the downloaded .vtt file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            logger.info("No native captions found.")
            return None

        vtt_text = vtt_files[0].read_text(encoding="utf-8")
        segments = _parse_vtt(vtt_text)
        logger.info(f"Parsed {len(segments)} caption segments from native captions.")
        return segments


# ─────────────────────────────────────────────
# Strategy B: Whisper transcription
# ─────────────────────────────────────────────

def fetch_whisper_transcript(url: str) -> list[TranscriptSegment]:
    """
    Download audio and transcribe with faster-whisper.
    Used as fallback when native captions are unavailable.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Download audio only
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--no-playlist",
            "-o", audio_path,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio download failed: {result.stderr}")

        logger.info(f"Transcribing with Whisper ({TRANSCRIPT.whisper_model})...")
        model = WhisperModel(
            TRANSCRIPT.whisper_model,
            device=TRANSCRIPT.whisper_device,
            compute_type=TRANSCRIPT.whisper_compute_type,
        )

        whisper_segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True,          # Skip silent sections
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        for seg in whisper_segments:
            text = seg.text.strip()
            if text:
                segments.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                ))

        logger.info(f"Whisper produced {len(segments)} segments. Language: {info.language}")
        return segments


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

@timed("stage1.transcript")
def get_transcript(url: str, force_whisper: bool = False) -> Transcript:
    """
    Full transcript generation pipeline for a YouTube URL.

    Strategy:
      1. Fetch video metadata (title, channel, date)
      2. Try native captions via yt-dlp (fast)
      3. Fall back to Whisper if no captions found or force_whisper=True

    Args:
        url: YouTube video URL
        force_whisper: Skip native caption check, always use Whisper

    Returns:
        Transcript object with all segments populated
    """
    Path(TRANSCRIPT.output_dir).mkdir(parents=True, exist_ok=True)

    # Check cache
    video_id = _extract_video_id(url)
    cache_path = Path(TRANSCRIPT.output_dir) / f"{video_id}.json"
    if cache_path.exists():
        logger.info(f"Loading cached transcript for {video_id}")
        return Transcript.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

    # Fetch metadata
    logger.info(f"Fetching metadata for: {url}")
    meta = _get_video_metadata(url)
    logger.info(f"Video: '{meta['video_title']}' by {meta['channel']}")

    # Get segments
    segments = None
    if not force_whisper:
        segments = fetch_native_captions(url, language=meta["language"])

    if segments is None:
        logger.info("Falling back to Whisper transcription...")
        segments = fetch_whisper_transcript(url)

    transcript = Transcript(
        video_id=meta["video_id"],
        video_title=meta["video_title"],
        channel=meta["channel"],
        video_url=meta["video_url"],
        published_date=meta["published_date"],
        language=meta["language"],
        segments=segments,
    )

    # Save to disk
    cache_path.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Transcript saved: {cache_path}")

    return transcript


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline/transcript.py <youtube_url> [--whisper]")
        sys.exit(1)

    url = sys.argv[1]
    force_whisper = "--whisper" in sys.argv
    t = get_transcript(url, force_whisper=force_whisper)
    print(f"\n✓ Transcript ready: {len(t.segments)} segments, {len(t.full_text)} chars")
    print(f"  Preview: {t.full_text[:300]}...")
