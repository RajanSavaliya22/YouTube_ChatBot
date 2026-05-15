"""
Shared data models used across all pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptSegment:
    """One timestamped segment from yt-dlp or Whisper."""
    start: float        # seconds
    end: float          # seconds
    text: str
    speaker: str = "unknown"


@dataclass
class Transcript:
    """Full transcript for one video."""
    video_id: str
    video_title: str
    channel: str
    video_url: str
    published_date: str
    language: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "channel": self.channel,
            "video_url": self.video_url,
            "published_date": self.published_date,
            "language": self.language,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segments = [TranscriptSegment(**s) for s in d.get("segments", [])]
        return cls(
            video_id=d["video_id"],
            video_title=d["video_title"],
            channel=d["channel"],
            video_url=d["video_url"],
            published_date=d["published_date"],
            language=d["language"],
            segments=segments,
        )


@dataclass
class ChunkPayload:
    """Every field stored alongside the vector in Qdrant."""
    video_id: str
    video_title: str
    channel: str
    video_url: str
    published_date: str
    language: str

    chunk_text: str        # Child chunk — what was embedded
    parent_text: str       # Parent chunk — what is sent to the LLM
    chunk_index: int       # Position among all chunks for this video

    timestamp_start: float
    timestamp_end: float
    timestamp_label: str   # e.g. "04:12 → 05:40"

    embedding_model: str = "BAAI/bge-large-en-v1.5"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkPayload":
        return cls(**d)


@dataclass
class RetrievedChunk:
    """Dense search result with score."""
    payload: ChunkPayload
    score: float
    chunk_id: str


def fmt_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
