"""
BM25 Tokenizer
===============
Converts raw text into tokens suitable for BM25 indexing.
Pipeline: lowercase → URL strip → punctuation remove → length filter
         → stopword removal → Porter stemming
"""

import re
import string
from functools import lru_cache

from config import BM25 as BM25_CFG
from utils.logger import get_logger

logger = get_logger("sparse.tokenizer")

# ─────────────────────────────────────────────
# NLTK setup (optional but recommended)
# ─────────────────────────────────────────────

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))
    _STEMMER = PorterStemmer()
    NLTK_AVAILABLE = True
except ImportError:
    _STOPWORDS = set()
    _STEMMER = None
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")

# Transcript-specific filler words to always remove
_TRANSCRIPT_NOISE = {
    "um", "uh", "er", "ah", "like", "okay", "right", "yeah", "know",
    "gonna", "wanna", "gotta", "kinda", "sorta", "basically", "literally",
    "actually", "obviously", "clearly", "really", "pretty", "quite",
    "just", "so", "well", "now", "also", "even", "still",
}

ALL_STOPWORDS = _STOPWORDS | _TRANSCRIPT_NOISE

# Compile patterns once at module load
_URL_RE = re.compile(r"http\S+|www\.\S+")
_TIMESTAMP_RE = re.compile(r"[\[\(]\d{1,2}:\d{2}[\]\)]")
_PUNCT_TABLE = str.maketrans(
    string.punctuation.replace("-", ""),
    " " * (len(string.punctuation) - 1),
)
_MULTI_SPACE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """
    Full tokenization pipeline for BM25 indexing.

    Steps:
      1. Lowercase
      2. Strip URLs and timestamp markers like [04:12]
      3. Remove punctuation (keep hyphens for compound words)
      4. Split on whitespace
      5. Filter by min/max token length
      6. Remove stopwords (NLTK English + transcript noise)
      7. Porter stem for recall improvement (running→run, taxes→tax)

    Returns:
        List of normalized tokens
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Strip URLs and timestamps
    text = _URL_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)

    # 3. Remove punctuation
    text = text.translate(_PUNCT_TABLE)

    # 4. Split
    tokens = _MULTI_SPACE.split(text.strip())

    # 5. Length filter
    tokens = [
        t for t in tokens
        if BM25_CFG.min_token_length <= len(t) <= BM25_CFG.max_token_length
    ]

    # 6. Stopwords
    if BM25_CFG.remove_stopwords:
        tokens = [t for t in tokens if t not in ALL_STOPWORDS]

    # 7. Stemming
    if BM25_CFG.stem and NLTK_AVAILABLE and _STEMMER:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


@lru_cache(maxsize=10_000)
def tokenize_cached(text: str) -> tuple[str, ...]:
    """
    Cached tokenization for queries (called repeatedly during search).
    Returns tuple for hashability.
    """
    return tuple(tokenize(text))
