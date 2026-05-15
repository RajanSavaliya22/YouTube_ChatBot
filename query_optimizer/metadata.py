"""
Technique 5: Metadata Filter Extraction
=========================================
Parse user queries for implicit scope constraints and convert them into
Qdrant payload filters — narrowing the search space before retrieval.

Without this: "What did Karpathy say about GPT-4?" searches ALL videos.
With this:    Detects "Karpathy" → sets filter_channel="Andrej Karpathy"
              → searches only Karpathy's videos → much higher precision.

Extracted filters:
  - channel:      "In Lex Fridman's video..." → filter_channel
  - video_id:     Explicit video ID in query  → filter_video_id
  - date_after:   "after 2023", "since 2024"  → filter_date_after (YYYYMMDD)
  - date_before:  "before 2022"               → filter_date_before
  - language:     "en" default

Two modes:
  A) Rule-based: regex patterns for dates/years (fast, no LLM needed)
  B) LLM-based:  extract channel names from natural language (when LLM available)

The extracted filters are passed directly to hybrid_search() as Qdrant
payload filters, reducing the search space before HNSW runs.
"""

import re
from dataclasses import dataclass, field
from query_optimizer.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger("query_optimizer.metadata")


@dataclass
class QueryFilters:
    """Extracted metadata filters to scope vector search."""
    channel: str | None = None
    video_id: str | None = None
    date_after: str | None = None    # YYYYMMDD format
    language: str = "en"

    def to_dict(self) -> dict:
        """Convert to kwargs for hybrid_search()."""
        out = {"filter_language": self.language}
        if self.channel:
            out["filter_channel"] = self.channel
        if self.video_id:
            out["filter_video_id"] = self.video_id
        if self.date_after:
            out["filter_date_after"] = self.date_after
        return out

    def is_empty(self) -> bool:
        return not any([self.channel, self.video_id, self.date_after])

    def __str__(self) -> str:
        parts = []
        if self.channel:
            parts.append(f"channel='{self.channel}'")
        if self.video_id:
            parts.append(f"video_id='{self.video_id}'")
        if self.date_after:
            parts.append(f"date_after={self.date_after}")
        return ", ".join(parts) if parts else "none"


# ── Rule-based date extraction ────────────────────────────────

_YEAR_AFTER = re.compile(
    r"(?:after|since|from|post[\-\s]?|starting\s+(?:from\s+)?)"
    r"(\d{4})",
    re.IGNORECASE,
)
_YEAR_IN = re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)
_YEAR_STANDALONE = re.compile(r"\b(20\d{2})\b")


def _extract_date_filter(query: str) -> str | None:
    """Extract a 'date after' filter from the query using regex."""
    m = _YEAR_AFTER.search(query)
    if m:
        return f"{m.group(1)}0101"

    m = _YEAR_IN.search(query)
    if m:
        return f"{m.group(1)}0101"

    # Fallback: if a single 4-digit year is mentioned, use it
    years = _YEAR_STANDALONE.findall(query)
    if len(years) == 1:
        return f"{years[0]}0101"

    return None


# ── LLM-based channel extraction ─────────────────────────────

_CHANNEL_SYSTEM = """You are a metadata extractor for a YouTube video search system.
Extract the YouTube channel or person name being referenced in the query.
Return ONLY the channel/person name, or "none" if no specific channel is mentioned.
No punctuation, no explanation."""

_CHANNEL_PROMPT = """Does this query reference a specific YouTube channel or content creator?
If yes, return their name. If no, return "none".

Query: {query}

Channel name (or "none"):"""


def _extract_channel_llm(query: str) -> str | None:
    """Use LLM to extract a channel name from the query."""
    response = call_llm(_CHANNEL_PROMPT.format(query=query), system=_CHANNEL_SYSTEM)
    if not response:
        return None
    cleaned = response.strip().strip('"').strip("'")
    if cleaned.lower() in ("none", "n/a", "no", "unknown", ""):
        return None
    # Sanity: channel name shouldn't be a full sentence
    if len(cleaned.split()) > 6:
        return None
    return cleaned


# ── Public API ────────────────────────────────────────────────

def extract_filters(
    query: str,
    explicit_channel: str | None = None,
    explicit_video_id: str | None = None,
) -> QueryFilters:
    """
    Extract metadata filters from a query.

    Priority order:
      1. Explicit CLI flags (--channel, --video) override everything
      2. LLM-extracted channel name from query text
      3. Rule-based date extraction from query text

    Args:
        query:            Raw user query
        explicit_channel: Channel filter from CLI (--channel=X)
        explicit_video_id: Video filter from CLI (--video=X)

    Returns:
        QueryFilters with all extracted constraints
    """
    filters = QueryFilters()

    # 1. Explicit overrides
    if explicit_channel:
        filters.channel = explicit_channel
    if explicit_video_id:
        filters.video_id = explicit_video_id

    # 2. LLM channel extraction (only if no explicit channel given)
    if not filters.channel:
        filters.channel = _extract_channel_llm(query)

    # 3. Rule-based date extraction
    filters.date_after = _extract_date_filter(query)

    if not filters.is_empty():
        logger.info(f"Metadata filters extracted: {filters}")
    else:
        logger.debug("No metadata filters extracted.")

    return filters