"""
Technique 2: HyDE — Hypothetical Document Embeddings
======================================================
Instead of embedding the user query directly, generate a hypothetical
"ideal answer" document and embed THAT for retrieval.

Why it works:
  Query embeddings and document embeddings live in different parts of the
  vector space (queries are short + interrogative, documents are long +
  declarative). HyDE bridges this gap by making the search vector look
  more like a real document chunk.

  Reference: Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without
  Relevance Labels" — https://arxiv.org/abs/2212.10496

Example:
  Query:  "How does attention work in transformers?"
  HyDE doc: "Attention in transformers works by computing query, key and value
             matrices... The softmax of QK^T / sqrt(d_k) gives attention weights..."

  The HyDE doc embedding is much closer to actual transformer explanation chunks
  than the raw question embedding.

Multiple docs: we generate `hyde_docs` (default 3) hypothetical documents,
embed each, and return the mean vector. This reduces variance from any single
LLM generation.

Fallback: returns None if LLM unavailable → caller uses original query vector.
"""

import numpy as np
from query_optimizer.llm_client import call_llm
from config import QUERY_OPTIMIZER
from utils.embedder import embed_texts
from utils.logger import get_logger

logger = get_logger("query_optimizer.hyde")

_SYSTEM = """You are a helpful assistant that generates example answers to questions.
Generate a concise, factual paragraph (100-150 words) that directly answers the question.
Write as if you are an expert explaining this in a YouTube video transcript.
Return ONLY the paragraph, no preamble or labels."""

_PROMPT = """Generate a hypothetical answer paragraph for this question:

Question: {query}

Answer paragraph:"""


def generate_hyde_vector(query: str) -> list[float] | None:
    """
    Generate HyDE embedding for a query.

    Steps:
      1. Generate `hyde_docs` hypothetical answer documents via LLM
      2. Embed each document
      3. Return the mean vector (reduces LLM generation variance)

    Args:
        query: User query string

    Returns:
        Mean embedding vector across all hypothetical docs,
        or None if LLM is unavailable
    """
    if not QUERY_OPTIMIZER.hyde_enabled:
        return None

    hypothetical_docs = []

    for i in range(QUERY_OPTIMIZER.hyde_docs):
        response = call_llm(_PROMPT.format(query=query), system=_SYSTEM)
        if response and len(response.split()) >= 20:  # Sanity: at least 20 words
            hypothetical_docs.append(response.strip())
        else:
            logger.debug(f"HyDE doc {i+1} rejected (too short or empty)")

    if not hypothetical_docs:
        logger.debug("HyDE skipped — no valid hypothetical documents generated.")
        return None

    logger.info(f"HyDE: generated {len(hypothetical_docs)} hypothetical docs for '{query[:50]}'")

    # Embed all docs and mean-pool
    embeddings = embed_texts(hypothetical_docs)
    mean_vec = np.mean(embeddings, axis=0)

    # Re-normalize (mean of L2-normalized vectors isn't necessarily unit length)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    return mean_vec.tolist()