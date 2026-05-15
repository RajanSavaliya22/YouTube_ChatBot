"""
Technique 3: Multi-Query Expansion
=====================================
Generate multiple paraphrases of the original query, retrieve for each
independently, then merge all results before fusion.

Why it helps:
  Embedding search is sensitive to exact phrasing. A slightly different
  wording can retrieve completely different (and relevant) chunks.
  By searching with 4 paraphrases + the original, we dramatically improve
  recall without sacrificing precision (RRF handles deduplication).

Example:
  Original: "how do neural networks learn?"
  Paraphrases:
    - "what is the training process for deep learning models?"
    - "backpropagation and gradient descent explained"
    - "how does a neural network update its weights?"
    - "machine learning model optimization techniques"

  Each retrieves slightly different chunks → merged pool is much richer.

Deduplication: handled downstream by RRF — chunks appearing across
multiple query retrievals get boosted scores naturally.

Fallback: returns [original_query] if LLM unavailable.
"""

import re
from query_optimizer.llm_client import call_llm
from config import QUERY_OPTIMIZER
from utils.logger import get_logger

logger = get_logger("query_optimizer.multiquery")

_SYSTEM = """You are a search query expansion assistant.
Generate alternative phrasings of the given search query to improve retrieval coverage.
Return ONLY a numbered list of alternative queries, one per line.
No explanations, no preamble."""

_PROMPT = """Generate {count} alternative phrasings of this search query.
Each should capture the same information need but with different wording.

Original query: {query}

{count} alternative phrasings:"""


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list response from the LLM into a Python list."""
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        # Strip leading numbers, dots, dashes, spaces
        cleaned = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
        cleaned = cleaned.strip('"').strip("'")
        if cleaned and len(cleaned) > 5:
            queries.append(cleaned)
    return queries


def expand_query(query: str) -> list[str]:
    """
    Generate multiple paraphrases of the query for parallel retrieval.

    Args:
        query: Original user query

    Returns:
        List starting with the original query, followed by paraphrases.
        Always contains at least [query] even on LLM failure.
    """
    if not QUERY_OPTIMIZER.multiquery_enabled:
        return [query]

    response = call_llm(
        _PROMPT.format(query=query, count=QUERY_OPTIMIZER.multiquery_count),
        system=_SYSTEM,
    )

    if not response:
        logger.debug("Multi-query expansion skipped — LLM unavailable.")
        return [query]

    paraphrases = _parse_numbered_list(response)

    # Filter: remove duplicates and anything too similar to original
    seen = {query.lower().strip()}
    unique = []
    for p in paraphrases:
        key = p.lower().strip()
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(p)

    # Always include original first
    all_queries = [query] + unique[:QUERY_OPTIMIZER.multiquery_count]

    logger.info(
        f"Multi-query: '{query[:40]}' → {len(all_queries)} queries total "
        f"(+{len(all_queries)-1} paraphrases)"
    )

    return all_queries