"""
Stage 1: Transcript Generation
================================
Fetches transcripts from YouTube videos using a three-tier strategy:

  Priority 1 — youtube-transcript-api (fast, no bot detection, no API key)
                Works on Render/cloud servers. Same caption quality as yt-dlp.

  Priority 2 — yt-dlp native captions (local only, blocked on cloud servers)
                Falls back to this when youtube-transcript-api fails locally.

  Priority 3 — Whisper (audio transcription, no captions needed)
                Last resort for videos with no captions at all.

Metadata (title, channel, date) fetched via:
  1. pytube  (no bot detection)
  2. yt-dlp  (fallback, local only)
  3. Minimal fallback using video ID

Output: Transcript JSON saved to storage/transcripts/{video_id}.json
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
# Shared helpers
# ─────────────────────────────────────────────

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any YouTube URL format."""
    patterns = [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"]
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


# ─────────────────────────────────────────────
# Metadata fetching
# ─────────────────────────────────────────────

def _get_metadata_pytube(url: str, video_id: str) -> dict | None:
    """
    Fetch video metadata via pytube.
    Works on cloud servers — no bot detection.
    """
    try:
        from pytubefix import YouTube
        yt = YouTube(url)
        title = yt.title or f"Video {video_id}"
        channel = yt.author or "Unknown"
        pub_date = ""
        if yt.publish_date:
            pub_date = yt.publish_date.strftime("%Y%m%d")

        logger.info(f"Metadata via pytube: '{title}' by {channel}")
        return {
            "video_id":       video_id,
            "video_title":    title,
            "channel":        channel,
            "video_url":      f"https://www.youtube.com/watch?v={video_id}",
            "published_date": pub_date,
            "language":       "en",
        }
    except Exception as e:
        logger.warning(f"pytube metadata failed: {e}")
        return None


def _get_metadata_ytdlp(url: str) -> dict | None:
    """
    Fetch video metadata via yt-dlp --dump-json (no download).
    Works locally; may be blocked on cloud servers.
    """
    try:
        cmd = [
            "yt-dlp", "--dump-json",
            "--no-playlist", "--skip-download", url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"yt-dlp metadata failed: {result.stderr[:200]}")
            return None
        info = json.loads(result.stdout)
        logger.info(f"Metadata via yt-dlp: '{info.get('title')}' by {info.get('uploader')}")
        return {
            "video_id":       info.get("id", ""),
            "video_title":    info.get("title", ""),
            "channel":        info.get("uploader", info.get("channel", "")),
            "video_url":      info.get("webpage_url", url),
            "published_date": info.get("upload_date", ""),
            "language":       info.get("language", "en") or "en",
        }
    except Exception as e:
        logger.warning(f"yt-dlp metadata failed: {e}")
        return None


def _get_video_metadata(url: str, video_id: str) -> dict:
    """
    Fetch metadata with fallback chain:
      pytube → yt-dlp → minimal (video ID only)
    """
    meta = _get_metadata_pytube(url, video_id)
    if meta:
        return meta

    meta = _get_metadata_ytdlp(url)
    if meta:
        return meta

    # Minimal fallback — just the video ID
    logger.warning(f"All metadata fetchers failed — using minimal fallback for {video_id}")
    return {
        "video_id":       video_id,
        "video_title":    f"Video {video_id}",
        "channel":        "Unknown",
        "video_url":      f"https://www.youtube.com/watch?v={video_id}",
        "published_date": "",
        "language":       "en",
    }


# ─────────────────────────────────────────────
# Webshare proxy configuration (works around YouTube's cloud-IP blocking)
# ─────────────────────────────────────────────

_ytt_client = None  # Singleton — reused across calls so we don't rebuild proxy config every time


def _get_ytt_client():
    """
    Build (or return cached) YouTubeTranscriptApi client.

    If WEBSHARE_PROXY_USERNAME / WEBSHARE_PROXY_PASSWORD are set, routes all
    requests through Webshare's residential proxy pool — required on Render
    and other cloud hosts, since YouTube blocklists most cloud provider IP
    ranges (AWS, GCP, Azure, Render, Railway, etc.) for transcript/caption
    requests, even though no API key or download is involved.

    Falls back to a direct (proxy-less) client if credentials are absent —
    works fine for local development on a residential IP.
    """
    global _ytt_client
    if _ytt_client is not None:
        return _ytt_client

    from youtube_transcript_api import YouTubeTranscriptApi

    proxy_user = os.getenv("WEBSHARE_PROXY_USERNAME")
    proxy_pass = os.getenv("WEBSHARE_PROXY_PASSWORD")

    if proxy_user and proxy_pass:
        from youtube_transcript_api.proxies import WebshareProxyConfig
        logger.info("youtube-transcript-api: using Webshare proxy (cloud IP workaround)")
        _ytt_client = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_user,
                proxy_password=proxy_pass,
            )
        )
    else:
        logger.info("youtube-transcript-api: no proxy configured — using direct connection")
        _ytt_client = YouTubeTranscriptApi()

    return _ytt_client


# ─────────────────────────────────────────────
# Strategy 1: youtube-transcript-api
# ─────────────────────────────────────────────

def fetch_youtube_transcript_api(
    video_id: str,
    language: str = "en",
) -> list[TranscriptSegment] | None:
    """
    Fetch captions via youtube-transcript-api (v1.0+ instance-based API).

    - No yt-dlp required
    - Works on Render/cloud servers when routed through a Webshare proxy
      (set WEBSHARE_PROXY_USERNAME / WEBSHARE_PROXY_PASSWORD env vars)
    - Same caption text as yt-dlp (same YouTube source)
    - Segment-level timestamps (3-5 second chunks)

    Returns None if no captions available for this video.
    """
    try:
        ytt_api = _get_ytt_client()
    except ImportError:
        logger.warning("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return None

    # Try preferred language first, then English variants, then any available
    lang_priority = [language, "en", "en-US", "en-GB", "en-CA"]
    lang_priority = list(dict.fromkeys(lang_priority))  # deduplicate

    try:
        transcript_list = ytt_api.list(video_id)

        # Try manually created captions first (higher quality than auto-generated)
        transcript = None
        for lang in lang_priority:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                logger.info(f"Found manual captions in '{lang}'")
                break
            except Exception:
                pass

        # Fall back to auto-generated
        if transcript is None:
            for lang in lang_priority:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    logger.info(f"Found auto-generated captions in '{lang}'")
                    break
                except Exception:
                    pass

        # Last resort: any available transcript
        if transcript is None:
            try:
                available = list(transcript_list)
                if available:
                    transcript = available[0]
                    logger.info(f"Using available transcript in '{transcript.language_code}'")
            except Exception:
                pass

        if transcript is None:
            logger.info("No transcripts found via youtube-transcript-api.")
            return None

        # v1.0+ fetch() returns a FetchedTranscript object with a `.snippets`
        # list of FetchedTranscriptSnippet objects (attribute access, not dict keys)
            fetched = _fetch_with_retry(transcript)        
            segments = [
            TranscriptSegment(
                start=float(snippet.start),
                end=float(snippet.start) + float(snippet.duration or 3.0),
                text=snippet.text.strip(),
            )
            for snippet in fetched.snippets
            if snippet.text.strip()
        ]

        logger.info(
            f"youtube-transcript-api: {len(segments)} segments "
            f"(manual={not transcript.is_generated})"
        )
        return segments

    except Exception as e:
        logger.warning(f"youtube-transcript-api failed: {e}")
        return None

    import time

    def _fetch_with_retry(transcript, max_retries=3, base_delay=2):
        for attempt in range(max_retries):
            try:
                return transcript.fetch()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"429 rate limit — retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
# ─────────────────────────────────────────────
# Strategy 2: yt-dlp captions (local fallback)
# ─────────────────────────────────────────────

def _parse_vtt(vtt_text: str) -> list[TranscriptSegment]:
    """Parse WebVTT caption file into TranscriptSegment list."""
    segments = []
    cue_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d+)\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d+)[^\n]*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    for match in cue_pattern.finditer(vtt_text):
        start_str, end_str, raw_text = match.groups()
        text = re.sub(r"<[^>]+>", "", raw_text).strip()
        text = re.sub(r"\n", " ", text).strip()
        if text:
            segments.append(TranscriptSegment(
                start=_seconds_to_float(start_str),
                end=_seconds_to_float(end_str),
                text=text,
            ))
    return segments


def fetch_ytdlp_captions(
    url: str,
    language: str = "en",
) -> list[TranscriptSegment] | None:
    """
    Download captions via yt-dlp.
    Works locally. Blocked on cloud servers by YouTube bot detection.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                "--write-auto-subs", "--write-subs",
                "--sub-lang", language,
                "--sub-format", "vtt",
                "--skip-download", "--no-playlist",
                "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                logger.info("yt-dlp: no caption files found.")
                return None
            vtt_text = vtt_files[0].read_text(encoding="utf-8")
            segments = _parse_vtt(vtt_text)
            logger.info(f"yt-dlp: parsed {len(segments)} caption segments.")
            return segments
    except Exception as e:
        logger.warning(f"yt-dlp captions failed: {e}")
        return None


# ─────────────────────────────────────────────
# Strategy 3: Whisper (no captions fallback)
# ─────────────────────────────────────────────

def fetch_whisper_transcript(url: str) -> list[TranscriptSegment]:
    """
    Download audio and transcribe with faster-whisper.
    Last resort for videos with no captions available.
    Requires yt-dlp for audio download — may fail on Render.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        cmd = [
            "yt-dlp", "--extract-audio",
            "--audio-format", "mp3", "--audio-quality", "0",
            "--no-playlist", "-o", audio_path, url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Audio download failed: {result.stderr[:300]}")

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
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in whisper_segments if seg.text.strip()
        ]
        logger.info(f"Whisper: {len(segments)} segments. Language detected: {info.language}")
        return segments


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

@timed("stage1.transcript")
def get_transcript(url: str, force_whisper: bool = False) -> Transcript:
    """
    Full transcript generation with three-tier fallback strategy.

    Order:
      1. youtube-transcript-api  — cloud-safe, no bot detection
      2. yt-dlp captions         — local fallback
      3. Whisper                 — for videos with no captions

    Args:
        url:           YouTube video URL
        force_whisper: Skip caption fetching, always use Whisper

    Returns:
        Transcript object with all segments populated
    """
    Path(TRANSCRIPT.output_dir).mkdir(parents=True, exist_ok=True)

    video_id = _extract_video_id(url)

    # Cache check
    cache_path = Path(TRANSCRIPT.output_dir) / f"{video_id}.json"
    if cache_path.exists():
        logger.info(f"Loading cached transcript for {video_id}")
        return Transcript.from_dict(json.loads(cache_path.read_text()))

    # Metadata
    logger.info(f"Fetching metadata for: {url}")
    meta = _get_video_metadata(url, video_id)
    logger.info(f"Video: '{meta['video_title']}' by {meta['channel']}")

    # Segments
    segments = None

    if not force_whisper:
        # Strategy 1: youtube-transcript-api (cloud-safe)
        logger.info("Trying youtube-transcript-api...")
        segments = fetch_youtube_transcript_api(video_id, language=meta["language"])

        # Strategy 2: yt-dlp (local fallback)
        if segments is None:
            logger.info("Trying yt-dlp captions...")
            segments = fetch_ytdlp_captions(url, language=meta["language"])

    # Strategy 3: Whisper
    if segments is None:
        logger.info("No captions found — falling back to Whisper...")
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

    cache_path.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2))
    logger.info(f"Transcript saved: {cache_path} ({len(segments)} segments)")

    return transcript


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline/01_transcript.py <youtube_url> [--whisper]")
        sys.exit(1)
    url = sys.argv[1]
    force_whisper = "--whisper" in sys.argv
    t = get_transcript(url, force_whisper=force_whisper)
    print(f"\n✓ {len(t.segments)} segments | {len(t.full_text)} chars")
    print(f"  Preview: {t.full_text[:300]}...")