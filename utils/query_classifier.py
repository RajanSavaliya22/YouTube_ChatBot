"""
Query Classifier
=================
Detects whether a user query is a "general/overview" question
(about the video as a whole) vs a "specific" question (about a detail).

Two-layer detection:
  Layer 1 — Regex patterns (instant, no model needed)
    Catches: "what is this video about", "summarize", "main topics", etc.

  Layer 2 — Embedding similarity (catches paraphrases)
    Embeds query → cosine sim against pre-embedded template questions
    Threshold: 0.75 (lower than semantic cache to be more inclusive)

General query examples:
  "What is discussed in this video?"
  "Summarize the key points"
  "What problems are solved?"
  "Give me an overview"
  "What are the main takeaways?"
  "What topics are covered?"

Specific query examples:
  "How does AlphaFold predict protein structure?"
  "What did the CEO say about scaling laws?"
  "When was the model released?"
"""

import re
from functools import lru_cache
from utils.logger import get_logger

logger = get_logger("utils.query_classifier")

# ── Layer 1: Regex patterns ───────────────────────────────────

_OVERVIEW_PATTERNS = re.compile(
    r"\b("
    r"what (is |are )?(this |the )?(video|interview|talk|episode|discussion|podcast)? ?(about|cover|discuss|explain|address|explore|focus)"
    r"|summar(y|ize|ise)"
    r"|overview"
    r"|main (point|topic|theme|idea|takeaway|concept|subject)"
    r"|key (point|topic|theme|idea|takeaway|concept|insight|lesson)"
    r"|what (topic|subject|issue|problem|concept|idea)s?"
    r"|what (problem|issue|challenge|question)s? (are|were|is|was) (solved|discussed|addressed|covered|explored|raised)"
    r"|what (does|did) (this|the) (video|talk|interview) (cover|discuss|explain|say|address)"
    r"|tell me about (this|the) (video|talk|interview|episode)"
    r"|give (me )?(an? )?(overview|summary|recap|rundown|breakdown)"
    r"|what happens in"
    r"|what (was|is) (talked|spoken|mentioned|said) (about|in)"
    r"|which (topic|subject|problem|concept|idea|theme)s?"
    r"|takeaways?"
    r"|overall (theme|message|point|content|idea)"
    r")\b",
    re.IGNORECASE,
)

# ── Layer 2: Embedding templates ──────────────────────────────

_OVERVIEW_TEMPLATES = [
    "What is this video about?",
    "Summarize this video for me",
    "What topics are covered in this video?",
    "What are the main points of this video?",
    "Give me an overview of this video",
    "What problems are discussed in this video?",
    "What are the key takeaways from this video?",
    "What does this video cover?",
    "Tell me about this video",
    "What is discussed in this video?",
]

_SIMILARITY_THRESHOLD = 0.75
_template_vector: list[float] | None = None   # Mean of all template embeddings


def _get_template_vector() -> list[float]:
    """Compute and cache the mean embedding of all overview templates."""
    global _template_vector
    if _template_vector is not None:
        return _template_vector

    try:
        import numpy as np
        from utils.embedder import embed_texts
        embeddings = embed_texts(_OVERVIEW_TEMPLATES)
        _template_vector = np.mean(embeddings, axis=0).tolist()
        logger.debug("Overview template vector computed and cached.")
    except Exception as e:
        logger.warning(f"Could not compute template vector: {e}")
        _template_vector = []

    return _template_vector


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Dot product of two L2-normalized vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def is_overview_query(query: str, query_vector: list[float] | None = None) -> bool:
    """
    Determine if a query is asking about the video overall.

    Args:
        query:        Raw user query string
        query_vector: Pre-computed embedding (reuses existing embed if available)

    Returns:
        True if the query is a general/overview question
    """
    # Layer 1: fast regex check
    if _OVERVIEW_PATTERNS.search(query):
        logger.info(f"Overview query detected [L1-regex]: '{query[:60]}'")
        return True

    # Layer 2: embedding similarity (only if vector provided or we can compute it)
    try:
        if query_vector is None:
            from utils.embedder import embed_query
            query_vector = embed_query(query)

        template_vec = _get_template_vector()
        if not template_vec:
            return False

        sim = _cosine_sim(query_vector, template_vec)
        logger.debug(f"Overview similarity: {sim:.3f} (threshold={_SIMILARITY_THRESHOLD})")

        if sim >= _SIMILARITY_THRESHOLD:
            logger.info(f"Overview query detected [L2-embed sim={sim:.3f}]: '{query[:60]}'")
            return True

    except Exception as e:
        logger.warning(f"Layer 2 classifier error: {e}")

    return False