"""
main.py — YouTube RAG Pipeline CLI
=====================================
Stages 1-8: transcript → clean → chunk → embed → vector store → cache
            → query optimization → re-ranking

  python main.py index  <youtube_url>   — Run full ingestion pipeline
  python main.py query  "<question>"    — Optimized cached hybrid search + rerank
  python main.py stats                  — Show all layer statistics
  python main.py cache-flush            — Flush all Redis cache keys
"""

import sys

from pipeline.transcript import get_transcript
from pipeline.cleaner import clean_transcript
from pipeline.chunker import chunk_transcript
from pipeline.embedder import embed_chunks

from vector_store.client import get_client, create_collection, get_collection_stats
from vector_store.indexer import upsert_chunks, delete_video, video_is_indexed
from vector_store.sparse.bm25_store import BM25Store

from cache.manager import CacheManager
from cache.embedding_cache import embed_with_cache

from query_optimizer.pipeline import QueryOptimizer
from query_optimizer.retriever import multi_query_retrieve

from reranker.model import rerank, RankedChunk

from utils.embedder import embed_texts
from utils.logger import get_logger

logger = get_logger("main")

# ── Singletons ────────────────────────────────────────────────

_bm25_store: BM25Store | None = None
_cache: CacheManager | None = None
_optimizer: QueryOptimizer | None = None


def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
        _bm25_store.load()
    return _bm25_store


def get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def get_optimizer() -> QueryOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer


# ── Index ─────────────────────────────────────────────────────

def cmd_index(url: str, force_reindex: bool = False, force_whisper: bool = False) -> None:
    """Run stages 1-5 for a YouTube URL. Stage 4 uses L3 embedding cache."""
    client = get_client()
    create_collection(client)
    bm25 = get_bm25_store()
    cache = get_cache()

    logger.info("=" * 50)
    logger.info(f"STAGE 1: Transcript — {url}")
    transcript = get_transcript(url, force_whisper=force_whisper)
    logger.info(f"  → {len(transcript.segments)} segments | video_id={transcript.video_id}")

    if not force_reindex and video_is_indexed(client, transcript.video_id):
        logger.info("Already indexed. Use --reindex to force.")
        return

    if force_reindex:
        delete_video(client, bm25, transcript.video_id)

    logger.info("STAGE 2: Cleaning")
    cleaned = clean_transcript(transcript)

    logger.info("STAGE 3: Chunking")
    chunks = chunk_transcript(cleaned)
    logger.info(f"  → {len(chunks)} chunks")
    if not chunks:
        logger.error("No chunks produced — aborting.")
        return

    logger.info("STAGE 4: Embedding  [L3 cache active]")
    texts = [c.chunk_text for c in chunks]
    embeddings = cache.embed(texts, embed_fn=embed_texts)
    logger.info(f"  → {len(embeddings)} vectors (dim={len(embeddings[0])})")

    logger.info("STAGE 5: Indexing into Qdrant + BM25")
    upsert_chunks(client, bm25, chunks, embeddings, save_bm25=True)

    stats = get_collection_stats(client)
    logger.info("=" * 50)
    logger.info(f"✓ Indexed '{transcript.video_title}'")
    logger.info(f"  Qdrant: {stats['total_vectors']} vectors | BM25: {bm25.stats()}")


# ── Query ─────────────────────────────────────────────────────

def cmd_query(
    question: str,
    top_k: int = 20,           # Retrieval candidates (more = better recall for reranker)
    top_n: int = 5,            # Final results after reranking
    channel: str | None = None,
    video_id: str | None = None,
    no_cache: bool = False,
    no_optimize: bool = False,
    no_rerank: bool = False,
) -> None:
    """
    Full query pipeline: cache → optimize → retrieve → rerank.

    Stage flow:
      [Cache]   L1 exact → L2 semantic           return immediately on hit
      [Stage 7] Rewrite → HyDE → multi-query → decompose → metadata filters
      [Stage 5] Hybrid retrieval: Qdrant ANN + BM25 + RRF  → top_k candidates
      [Stage 8] Cross-encoder re-ranking                    → top_n final results
      [Cache]   Store result in L1 + L2
    """
    client = get_client()
    bm25 = get_bm25_store()
    cache = get_cache()
    optimizer = get_optimizer()

    # ── Cache lookup (L1 + L2) ────────────────────────────────
    from utils.embedder import embed_query
    raw_vec = embed_query(question)
    filters_explicit = {k: v for k, v in {"channel": channel, "video_id": video_id}.items() if v}

    if not no_cache:
        cached = cache.get_query(question, raw_vec, **filters_explicit)
        if cached is not None:
            _print_results(question, cached, from_cache=True)
            return

    # ── Stage 7: Query Optimization ──────────────────────────
    if no_optimize:
        from vector_store.retriever import hybrid_search
        from query_optimizer.metadata import QueryFilters
        fused = hybrid_search(
            client=client,
            bm25_store=bm25,
            query_vector=raw_vec,
            query_text=question,
            top_k=top_k,
            **QueryFilters(channel=channel, video_id=video_id).to_dict(),
        )
        rewrite_query = question
    else:
        logger.info("STAGE 7: Query Optimization")
        optimized = optimizer.run(
            question,
            explicit_channel=channel,
            explicit_video_id=video_id,
        )
        logger.info("Retrieving with optimized query variants...")
        fused = multi_query_retrieve(
            client=client,
            bm25_store=bm25,
            optimized=optimized,
            top_k=top_k,
        )
        rewrite_query = optimized.rewritten

    # ── Stage 8: Re-ranking ───────────────────────────────────
    if no_rerank:
        # Fallback: no reranking, return top_n by RRF score
        result_data = [_fused_to_dict(r) for r in fused[:top_n]]
    else:
        logger.info(f"STAGE 8: Re-ranking {len(fused)} candidates → top {top_n}")
        ranked = rerank(
            query=rewrite_query,
            candidates=fused,
            top_n=top_n,
        )
        result_data = [_ranked_to_dict(r) for r in ranked]

    # ── Cache result ──────────────────────────────────────────
    if not no_cache:
        cache.set_query(question, raw_vec, result_data, **filters_explicit)

    _print_results(question, result_data, from_cache=False)


# ── Serialization helpers ─────────────────────────────────────

def _fused_to_dict(r) -> dict:
    """Serialize a FusedResult for caching and display."""
    return {
        "chunk_id":      r.chunk_id,
        "rerank_score":  None,
        "retrieval_rrf": r.rrf_score,
        "retrieval_rank": None,
        "source":        r.source,
        "payload":       r.payload.to_dict(),
    }


def _ranked_to_dict(r: RankedChunk) -> dict:
    """Serialize a RankedChunk for caching and display."""
    return {
        "chunk_id":      r.chunk_id,
        "rerank_score":  round(r.rerank_score, 4),
        "retrieval_rrf": round(r.retrieval_rrf, 4),
        "retrieval_rank": r.retrieval_rank,
        "source":        "reranked",
        "payload":       r.payload.to_dict(),
    }


def _print_results(question: str, results: list[dict], from_cache: bool) -> None:
    cache_tag = " [CACHED]" if from_cache else ""
    print(f"\n{'='*60}")
    print(f"Query: {question}{cache_tag}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        p = r["payload"]
        rerank_score = r.get("rerank_score")
        retrieval_rank = r.get("retrieval_rank")

        score_str = (
            f"rerank={rerank_score:.3f} (was #{retrieval_rank})"
            if rerank_score is not None
            else f"RRF={r.get('retrieval_rrf', 0.0):.4f}"
        )

        print(f"#{i} [{r.get('source', 'unknown').upper()}] {score_str}")
        print(f"   Video:   {p['video_title']}")
        print(f"   Channel: {p['channel']}")
        print(f"   Time:    {p['timestamp_label']}  →  {p['video_url']}?t={int(p['timestamp_start'])}")
        print(f"   Text:    {p['chunk_text'][:200]}...")
        print()


# ── Stats ─────────────────────────────────────────────────────

def cmd_stats() -> None:
    from query_optimizer.llm_client import is_available as llm_ok
    from config import QUERY_OPTIMIZER, RERANKER

    client = get_client()
    bm25 = get_bm25_store()
    cache = get_cache()

    print("\n── Qdrant ───────────────────────────────")
    for k, v in get_collection_stats(client).items():
        print(f"  {k}: {v}")

    print("\n── BM25 ─────────────────────────────────")
    for k, v in bm25.stats().items():
        print(f"  {k}: {v}")

    print("\n── Cache ────────────────────────────────")
    for k, v in cache.stats().items():
        print(f"  {k}: {v}")

    print("\n── Query Optimizer ──────────────────────")
    print(f"  LLM available: {llm_ok()}")
    print(f"  model:         {QUERY_OPTIMIZER.llm_model}")
    print(f"  rewrite:       {QUERY_OPTIMIZER.rewrite_enabled}")
    print(f"  hyde:          {QUERY_OPTIMIZER.hyde_enabled}")
    print(f"  multi-query:   {QUERY_OPTIMIZER.multiquery_enabled}")
    print(f"  decompose:     {QUERY_OPTIMIZER.decompose_enabled}")

    print("\n── Re-ranker ────────────────────────────")
    print(f"  model:     {RERANKER.model_name}")
    print(f"  device:    {RERANKER.device}")
    print(f"  top_n:     {RERANKER.top_n}")
    print(f"  enabled:   {RERANKER.enabled}")
    print(f"  threshold: {RERANKER.score_threshold}")
    print()


def cmd_cache_flush() -> None:
    deleted = get_cache().flush()
    print(f"✓ Flushed {deleted} cache keys.")


# ── CLI ───────────────────────────────────────────────────────

def print_help():
    print("""
YouTube RAG Pipeline — Stages 1-8

Usage:
  python main.py index       <youtube_url>  [--reindex] [--whisper]
  python main.py query       "<question>"   [--top_k=20] [--top_n=5]
                                            [--channel=X] [--video=ID]
                                            [--no-cache] [--no-optimize] [--no-rerank]
  python main.py stats
  python main.py cache-flush

Flags (query):
  --top_k=N     Retrieval candidates before reranking (default: 20)
  --top_n=N     Final results after reranking (default: 5)
  --no-rerank   Skip re-ranker, return top_n by RRF score
  --no-optimize Skip Stage 7 query optimization
  --no-cache    Bypass cache for this query
""")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print_help(); sys.exit(0)

    command = args[0]

    if command == "index":
        if len(args) < 2:
            print("Error: provide a YouTube URL"); sys.exit(1)
        cmd_index(
            args[1],
            force_reindex="--reindex" in args,
            force_whisper="--whisper" in args,
        )

    elif command == "query":
        if len(args) < 2:
            print("Error: provide a question"); sys.exit(1)
        cmd_query(
            question=args[1],
            top_k=next((int(a.split("=")[1]) for a in args if a.startswith("--top_k=")), 20),
            top_n=next((int(a.split("=")[1]) for a in args if a.startswith("--top_n=")), 5),
            channel=next((a.split("=")[1] for a in args if a.startswith("--channel=")), None),
            video_id=next((a.split("=")[1] for a in args if a.startswith("--video=")), None),
            no_cache="--no-cache" in args,
            no_optimize="--no-optimize" in args,
            no_rerank="--no-rerank" in args,
        )

    elif command == "stats":
        cmd_stats()

    elif command == "cache-flush":
        cmd_cache_flush()

    else:
        print(f"Unknown command: {command}"); print_help(); sys.exit(1)