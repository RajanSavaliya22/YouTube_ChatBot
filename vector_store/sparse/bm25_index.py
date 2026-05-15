"""
BM25 Scoring Engine
====================
Pure Python implementation of BM25+ over an inverted index.
Designed for in-memory use with up to ~1M chunks comfortably.

BM25 formula:
  score(q, d) = Σ IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))

where IDF = log((N - df + 0.5) / (df + 0.5) + 1), floored at epsilon.
"""

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from vector_store.sparse.tokenizer import tokenize
from config import BM25 as BM25_CFG
from utils.logger import get_logger

logger = get_logger("sparse.bm25_index")


@dataclass
class BM25Index:
    """
    In-memory BM25 index.

    Data structures:
      doc_term_freqs:  chunk_id → Counter of term frequencies
      inverted_index:  term → set of chunk_ids containing that term
      doc_lengths:     chunk_id → token count
    """

    doc_term_freqs: dict[str, Counter] = field(default_factory=dict)
    inverted_index: dict[str, set]     = field(default_factory=lambda: defaultdict(set))
    doc_lengths:    dict[str, int]     = field(default_factory=dict)

    avg_doc_length: float = 0.0
    num_docs:       int   = 0

    # ─── Indexing ─────────────────────────────────────────────────

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        """
        Index a batch of documents.
        Can be called incrementally — new docs are added without rebuilding.
        """
        added = 0
        for chunk_id, text in zip(chunk_ids, texts):
            tokens = tokenize(text)
            if not tokens:
                continue

            tf = Counter(tokens)
            self.doc_term_freqs[chunk_id] = tf
            self.doc_lengths[chunk_id] = len(tokens)

            for term in tf:
                self.inverted_index[term].add(chunk_id)

            added += 1

        self._recompute_stats()
        logger.debug(f"Added {added} docs. Total: {self.num_docs}")

    def remove_documents(self, chunk_ids: list[str]) -> None:
        """
        Remove documents by ID.
        Called when a video is deleted or re-indexed.
        """
        for chunk_id in chunk_ids:
            if chunk_id not in self.doc_term_freqs:
                continue

            # Remove from inverted index
            for term in self.doc_term_freqs[chunk_id]:
                self.inverted_index[term].discard(chunk_id)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

            del self.doc_term_freqs[chunk_id]
            del self.doc_lengths[chunk_id]

        self._recompute_stats()

    # ─── Scoring ──────────────────────────────────────────────────

    def score(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        BM25 score a query against all indexed documents.

        Efficiency: only scores documents that share at least one token
        with the query (skips irrelevant docs entirely).

        Args:
            query: Raw query string (will be tokenized internally)
            top_k: Maximum results to return

        Returns:
            List of (chunk_id, bm25_score) sorted descending
        """
        if self.num_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            candidate_ids = self.inverted_index[term]
            df = len(candidate_ids)

            # IDF with epsilon floor (prevents negative scores for very common terms)
            idf = max(
                BM25_CFG.epsilon,
                math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1),
            )

            for chunk_id in candidate_ids:
                tf = self.doc_term_freqs[chunk_id][term]
                doc_len = self.doc_lengths[chunk_id]

                # BM25 normalized TF
                tf_norm = (
                    tf * (BM25_CFG.k1 + 1)
                ) / (
                    tf + BM25_CFG.k1 * (
                        1 - BM25_CFG.b + BM25_CFG.b * (doc_len / self.avg_doc_length)
                    )
                )

                scores[chunk_id] += idf * tf_norm

        # Sort and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ─── Stats ────────────────────────────────────────────────────

    def _recompute_stats(self) -> None:
        self.num_docs = len(self.doc_lengths)
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
        else:
            self.avg_doc_length = 0.0

    def stats(self) -> dict:
        return {
            "num_docs": self.num_docs,
            "vocabulary_size": len(self.inverted_index),
            "avg_doc_length_tokens": round(self.avg_doc_length, 1),
        }
