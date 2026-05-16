"""
main.py — YouTube RAG Pipeline CLI
=====================================
Stages 1-9: transcript → clean → chunk → embed → vector store → cache
            → query optimization → re-ranking → LLM generation

  python main.py index  <youtube_url>   — Run full ingestion pipeline
  python main.py query  "<question>"    — Full RAG query with streaming answer
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
from generator.pipeline import Generator, GenerationResult
from generator.transcript_context import build_transcript_context
from generator.overview_prompt import build_overview_prompts
from generator.llm import generate

from utils.embedder import embed_texts
from utils.query_classifier import is_overview_query
from utils.logger import get_logger

logger = get_logger("main")

# ── Singletons ────────────────────────────────────────────────

_bm25_store: BM25Store | None = None
_cache: CacheManager | None = None
_optimizer: QueryOptimizer | None = None
_generator: Generator | None = None


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


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


# ── Index ─────────────────────────────────────────────────────

def cmd_index(url: str, force_reindex: bool = False, force_whisper: bool = False) -> None:
    """Run stages 1-5 for a YouTube URL."""
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

# ── Overview Route ────────────────────────────────────────────

def cmd_overview(
    question: str,
    video_id: str | None = None,
    channel: str | None = None,
    no_cache: bool = False,
    stream: bool = True,
) -> None:
    """
    Overview/summary route for general questions about a video.

    Instead of retrieving specific chunks, this:
      1. Samples chunks evenly across the whole video (breadth > precision)
      2. Feeds sampled transcript as context to the LLM
      3. Uses a different system prompt tuned for synthesis/summary

    Best for:
      "What is this video about?"
      "What topics are covered?"
      "What problems are solved?"
      "Summarize the key points"
    """
    bm25 = get_bm25_store()
    cache = get_cache()

    # ── Cache lookup ──────────────────────────────────────────
    from utils.embedder import embed_query
    raw_vec = embed_query(question)
    cache_key = f"overview:{question}"
    filters = {k: v for k, v in {"video_id": video_id, "channel": channel}.items() if v}

    if not no_cache:
        cached = cache.get_query(cache_key, raw_vec, **filters)
        if cached is not None:
            _print_cached_result(question, cached)
            return

    # ── Build transcript context ──────────────────────────────
    logger.info("Overview route: building transcript context...")
    ctx = build_transcript_context(bm25, video_id=video_id, channel=channel)

    if ctx is None:
        print(
            "\nNo indexed videos found. "
            + (f"No chunks for video_id='{video_id}'." if video_id else "")
            + "\nIndex a video first with: python main.py index <url>"
        )
        return

    logger.info(
        f"  '{ctx.video_title}' | "
        f"{ctx.chunks_used}/{ctx.total_chunks_available} chunks sampled"
    )

    # ── Build prompt ──────────────────────────────────────────
    system_prompt, user_message = build_overview_prompts(question, ctx)

    # ── Generate ──────────────────────────────────────────────
    logger.info("Generating overview answer...")
    print(f"\n{'='*60}")
    print(f"Q: {question}  [OVERVIEW — {ctx.chunks_used} segments sampled]")
    print(f"{'='*60}\n")

    full_answer = ""
    if stream:
        for token in generate(system_prompt, user_message, stream=True):
            print(token, end="", flush=True)
            full_answer += token
        print()
    else:
        full_answer = generate(system_prompt, user_message, stream=False)
        print(full_answer)

    # Print video info footer
    print(f"\n── Video ────────────────────────────────────────────")
    print(f"   {ctx.video_title}")
    print(f"   {ctx.channel}  →  {ctx.video_url}")

    # ── Cache result ──────────────────────────────────────────
    if not no_cache and full_answer:
        cache.set_query(cache_key, raw_vec, {
            "answer": full_answer,
            "citations": [],
            "is_overview": True,
            "video_title": ctx.video_title,
            "channel": ctx.channel,
        }, **filters)


# ── Query ─────────────────────────────────────────────────────

def cmd_query(
    question: str,
    top_k: int = 20,
    top_n: int = 5,
    channel: str | None = None,
    video_id: str | None = None,
    no_cache: bool = False,
    no_optimize: bool = False,
    no_rerank: bool = False,
    no_generate: bool = False,
    stream: bool = True,
    force_overview: bool = False,
    force_specific: bool = False,
) -> None:
    """
    Full RAG pipeline: cache → optimize → retrieve → rerank → generate.

    Stage flow:
      [Cache]   L1 exact → L2 semantic              → return immediately on hit
      [Stage 7] Rewrite → HyDE → multi-query → decompose → metadata filters
      [Stage 5] Qdrant ANN + BM25 + RRF              → top_k candidates
      [Stage 8] Cross-encoder re-ranking             → top_n chunks
      [Stage 9] LLM generation with citations        → streamed answer
      [Cache]   Store result in L1 + L2
    
    Auto-routing query entry point.

    Routes to overview if the question is detected as general,
    otherwise runs the full RAG pipeline (stages 5-9).

    Override routing with --overview or --specific flags.
    """
        # ── Auto-routing ──────────────────────────────────────────
    if not force_specific:
        from utils.embedder import embed_query
        raw_vec = embed_query(question)

        if force_overview or is_overview_query(question, raw_vec):
            logger.info("Routing to OVERVIEW pipeline")
            cmd_overview(
                question=question,
                video_id=video_id,
                channel=channel,
                no_cache=no_cache,
                stream=stream,
            )
            return

    logger.info("Routing to SPECIFIC RAG pipeline")


    client = get_client()
    bm25 = get_bm25_store()
    cache = get_cache()
    optimizer = get_optimizer()
    generator = get_generator()

    # ── Cache lookup ──────────────────────────────────────────
    from utils.embedder import embed_query
    raw_vec = embed_query(question)
    filters_explicit = {
        k: v for k, v in {"channel": channel, "video_id": video_id}.items() if v
    }

    if not no_cache:
        cached = cache.get_query(question, raw_vec, **filters_explicit)
        if cached is not None:
            _print_cached_result(question, cached)
            return

    # ── Stage 7: Query Optimization ──────────────────────────
    if no_optimize:
        from vector_store.retriever import hybrid_search
        from query_optimizer.metadata import QueryFilters
        fused = hybrid_search(
            client=client, bm25_store=bm25,
            query_vector=raw_vec, query_text=question,
            top_k=top_k,
            **QueryFilters(channel=channel, video_id=video_id).to_dict(),
        )
        rewritten_query = question
    else:
        logger.info("STAGE 7: Query Optimization")
        optimized = optimizer.run(
            question,
            explicit_channel=channel,
            explicit_video_id=video_id,
        )
        fused = multi_query_retrieve(
            client=client, bm25_store=bm25,
            optimized=optimized, top_k=top_k,
        )
        rewritten_query = optimized.rewritten

    # ── Stage 8: Re-ranking ───────────────────────────────────
    if no_rerank:
        from reranker.model import _fallback
        ranked = _fallback(fused, top_n)
    else:
        logger.info(f"STAGE 8: Re-ranking {len(fused)} candidates → top {top_n}")
        ranked = rerank(query=rewritten_query, candidates=fused, top_n=top_n)

    # ── Stage 9: LLM Generation ───────────────────────────────
    if no_generate:
        # Return raw retrieval results without LLM
        _print_raw_results(question, ranked)
        return

    logger.info("STAGE 9: Generating answer...")
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}\n")

    if stream:
        # Streaming: print tokens as they arrive
        token_gen, result = generator.stream(rewritten_query, ranked)
        full_answer = ""
        for token in token_gen:
            print(token, end="", flush=True)
            full_answer += token
        result.answer = full_answer
        print()  # Newline after streaming completes
    else:
        result = generator.run(rewritten_query, ranked)
        print(result.answer)

    # Print citations footer
    print(result.format_citations())

    # ── Cache result ──────────────────────────────────────────
    if not no_cache:
        cached_payload = {
            "answer":     result.answer,
            "citations":  [
                {
                    "index":              c.index,
                    "video_title":        c.video_title,
                    "channel":            c.channel,
                    "timestamp_label":    c.timestamp_label,
                    "url_with_timestamp": c.url_with_timestamp,
                }
                for c in result.citations
            ],
            "is_confident":       result.is_confident,
            "context_chunks_used": result.context_chunks_used,
        }
        cache.set_query(question, raw_vec, cached_payload, **filters_explicit)


# ── Display helpers ───────────────────────────────────────────

def _print_cached_result(question: str, cached: dict) -> None:
    """Display a cached answer (already has answer + citations)."""
    print(f"\n{'='*60}")
    print(f"Q: {question}  [CACHED]")
    print(f"{'='*60}\n")

    if "answer" in cached:
        # Full generated answer cached
        print(cached["answer"])
        if cached.get("citations"):
            print("\n── Sources ──────────────────────────────────────────")
            for c in cached["citations"]:
                print(
                    f"[Source {c['index']}] {c['video_title']} | {c['channel']}\n"
                    f"           {c['timestamp_label']} → {c['url_with_timestamp']}"
                )
    else:
        # Legacy: raw retrieval results cached (no generate step)
        _print_raw_results(question, cached)


def _print_raw_results(question: str, ranked) -> None:
    """Display raw re-ranked chunks (no LLM answer)."""
    print(f"\n{'='*60}")
    print(f"Q: {question}  [RAW RETRIEVAL — no LLM]")
    print(f"{'='*60}\n")
    for i, r in enumerate(ranked, 1):
        p = r.payload if hasattr(r, "payload") else r.get("payload", {})
        if hasattr(p, "video_title"):
            print(f"#{i} rerank={r.rerank_score:.3f} (was #{r.retrieval_rank})")
            print(f"   {p.video_title} | {p.channel}")
            print(f"   {p.timestamp_label} → {p.video_url}?t={int(p.timestamp_start)}")
            print(f"   {p.chunk_text[:200]}...")
        print()

def _print_raw_results(question: str, ranked) -> None:
    print(f"\n{'='*60}")
    print(f"Q: {question}  [RAW RETRIEVAL — no LLM]")
    print(f"{'='*60}\n")
    for i, r in enumerate(ranked, 1):
        p = r.payload
        print(f"#{i} rerank={r.rerank_score:.3f} (was #{r.retrieval_rank})")
        print(f"   {p.video_title} | {p.channel}")
        print(f"   {p.timestamp_label} → {p.video_url}?t={int(p.timestamp_start)}")
        print(f"   {p.chunk_text[:200]}...")
        print()

# ── Stats ─────────────────────────────────────────────────────

def cmd_stats() -> None:
    from query_optimizer.llm_client import is_available as llm_ok
    from config import QUERY_OPTIMIZER, RERANKER, GENERATOR

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
    print(f"  LLM:         {llm_ok()}")
    print(f"  model:       {QUERY_OPTIMIZER.llm_model}")
    print(f"  rewrite:     {QUERY_OPTIMIZER.rewrite_enabled}")
    print(f"  hyde:        {QUERY_OPTIMIZER.hyde_enabled}")
    print(f"  multi-query: {QUERY_OPTIMIZER.multiquery_enabled}")
    print(f"  decompose:   {QUERY_OPTIMIZER.decompose_enabled}")

    print("\n── Re-ranker ────────────────────────────")
    print(f"  model:     {RERANKER.model_name}")
    print(f"  top_n:     {RERANKER.top_n}")
    print(f"  enabled:   {RERANKER.enabled}")

    print("\n── Generator ────────────────────────────")
    print(f"  model:       {GENERATOR.model}")
    print(f"  stream:      {GENERATOR.stream}")
    print(f"  max_tokens:  {GENERATOR.max_tokens}")
    print(f"  temperature: {GENERATOR.temperature}")
    print(f"  max_chunks:  {GENERATOR.max_context_chunks}")
    print()


def cmd_cache_flush() -> None:
    deleted = get_cache().flush()
    print(f"✓ Flushed {deleted} cache keys.")


# ── CLI ───────────────────────────────────────────────────────

def print_help():
    print("""
YouTube RAG Chatbot — Stages 1-9

Usage:
  python main.py index  <youtube_url>  [--reindex] [--whisper]
  python main.py query  "<question>"   [--top_k=20] [--top_n=5]
                                       [--channel=X] [--video=ID]
                                       [--no-cache] [--no-optimize]
                                       [--no-rerank] [--no-generate]
                                       [--no-stream]
  python main.py overview "<question>"   [--video=ID] [--channel=X]
                                         [--no-cache] [--no-stream]
  python main.py stats
  python main.py cache-flush

Routing (query command):
  Auto-detects overview vs specific questions.
  --overview   Force overview route (full transcript context)
  --specific   Force specific RAG route (chunk retrieval)

Overview route is used for:
  "What is this video about?", "Summarize", "What topics are covered?"
  "What problems are solved?", "Main takeaways", etc.
Query flags:
  --top_k=N      Retrieval candidates before reranking (default: 20)
  --top_n=N      Chunks sent to LLM after reranking (default: 5)
  --no-stream    Return full answer at once instead of streaming
  --no-generate  Skip LLM, show raw retrieval results only
  --no-rerank    Skip Stage 8 re-ranker
  --no-optimize  Skip Stage 7 query optimization
  --no-cache     Bypass cache for this query
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
            no_generate="--no-generate" in args,
            stream="--no-stream" not in args,
            force_overview="--overview" in args,
            force_specific="--specific" in args,
        )

    elif command == "overview":
        if len(args) < 2:
            print("Error: provide a question"); sys.exit(1)
        cmd_overview(
            question=args[1],
            video_id=next((a.split("=")[1] for a in args if a.startswith("--video=")), None),
            channel=next((a.split("=")[1] for a in args if a.startswith("--channel=")), None),
            no_cache="--no-cache" in args,
            stream="--no-stream" not in args,
        )

    elif command == "stats":
        cmd_stats()

    elif command == "cache-flush":
        cmd_cache_flush()

    else:
        print(f"Unknown command: {command}"); print_help(); sys.exit(1)