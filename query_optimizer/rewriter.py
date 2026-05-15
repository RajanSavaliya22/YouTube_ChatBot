"""
Technique 1: Query Rewriting
==============================
Rewrites a vague or conversational user query into a retrieval-optimized form.

Problem it solves:
  User: "what did he say about taxes?"
  → Too vague, missing context, bad for embedding search

After rewrite:
  → "tax implications for content creators explained"
  → More specific, noun-heavy, retrieval-friendly

When it helps:
  - Conversational queries with pronouns ("he", "they", "it")
  - Vague queries missing domain context
  - Questions phrased as chat, not search

Fallback: returns the original query unchanged if LLM is unavailable.
"""

from query_optimizer.llm_client import call_llm
from config import QUERY_OPTIMIZER
from utils.logger import get_logger

logger = get_logger("query_optimizer.rewriter")

_SYSTEM = """You are a search query optimizer for a YouTube video knowledge base.
Your job is to rewrite user queries to improve semantic search retrieval.

Rules:
- Make the query specific and noun-heavy (good for embedding search)
- Remove vague pronouns (he, they, it) and replace with the actual subject if inferable
- Expand abbreviations
- Remove filler words
- Return ONLY the rewritten query, nothing else — no explanation, no prefix
- If the query is already good, return it unchanged
- NEVER expand or interpret acronyms — keep them exactly as written
- NEVER change the meaning or topic of the query
"""

_PROMPT = """Rewrite this search query to be more retrieval-friendly:

Original: {query}

Rewritten:"""


def rewrite_query(query: str) -> str:
    if not QUERY_OPTIMIZER.rewrite_enabled:
        return query

    response = call_llm(_PROMPT.format(query=query), system=_SYSTEM)
    if not response:
        return query

    rewritten = response.strip().strip('"').strip("'")
    if not rewritten or len(rewritten) > len(query) * 4:
        return query

    # Reject if rewritten query is semantically too different from original
    from utils.embedder import embed_query
    import numpy as np
    orig_vec = np.array(embed_query(query))
    new_vec  = np.array(embed_query(rewritten))
    similarity = float(np.dot(orig_vec, new_vec))  # already L2-normalized

    if similarity < 0.85:   # Threshold — tune between 0.80–0.90
        logger.warning(
            f"Rewrite rejected (similarity={similarity:.3f}): '{rewritten[:60]}'"
        )
        return query

    logger.info(f"Query rewrite (sim={similarity:.3f}): '{query}' → '{rewritten}'")
    return rewritten