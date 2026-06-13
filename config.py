"""
Central configuration for the YouTube RAG pipeline.
All settings are read from environment variables with sane defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Stage 1: Transcript ───────────────────────────────────────


@dataclass
class TranscriptConfig:
    output_dir: str = os.getenv("TRANSCRIPTS_PATH", "storage/transcripts")
    cleaned_dir: str = os.getenv("CLEANED_PATH", "storage/cleaned")
    chunks_dir: str = os.getenv("CHUNKS_PATH", "storage/chunks")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium/large
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE", "int8")

# ── Stage 3: Chunking ─────────────────────────────────────────

@dataclass
class ChunkingConfig:
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "300"))   # tokens
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "1200"))  # tokens
    overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Stage 4: Embeddings ───────────────────────────────────────

 
@dataclass
class EmbeddingConfig:
    # Backend: "local" (sentence-transformers) | "groq" (API, no RAM cost)
    backend: str    = os.getenv("EMBEDDER_BACKEND", "local")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    device: str     = os.getenv("EMBED_DEVICE", "cpu")
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Voyage AI embedding settings
    voyage_model: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-lite")
    voyage_batch: int = int(os.getenv("VOYAGE_EMBED_BATCH", "128"))
# ── Stage 5: Vector Store — Qdrant ───────────────────────────

@dataclass
class QdrantConfig:
    url: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    use_grpc: bool = os.getenv("QDRANT_USE_GRPC", "true").lower() == "true"
    api_key: str | None = os.getenv("QDRANT_API_KEY", None)


@dataclass
class CollectionConfig:
    name: str = os.getenv("COLLECTION_NAME", "youtube_chunks")
    vector_size: int = int(os.getenv("VECTOR_SIZE", "1024"))  # bge-large-en-v1.5 = 1024
    on_disk_payload: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    quantization: bool = True


# ── Stage 5: Vector Store — BM25 sparse ──────────────────────

@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    index_path: str = os.getenv("BM25_INDEX_PATH", "storage/bm25_index")
    index_file: str = "bm25.pkl"
    min_token_length: int = 2
    max_token_length: int = 40
    remove_stopwords: bool = True
    stem: bool = True

# ── Stage 5: Hybrid Retrieval — RRF fusion ───────────────────

@dataclass
class FusionConfig:
    rrf_k: int = 60
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 20

# ── Stage 6: Caching — Redis ─────────────────────────────────

@dataclass
class CacheConfig:
    # Redis connection
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str | None = os.getenv("REDIS_PASSWORD", None)
 
    # L1 — Exact cache: normalized query hash → full answer JSON
    exact_ttl: int = int(os.getenv("EXACT_CACHE_TTL", "3600"))          # 1 hour
    exact_prefix: str = "rag:exact:"
 
    # L2 — Semantic cache: query embedding similarity → cached answer
    semantic_ttl: int = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))   # 24 hours
    semantic_prefix: str = "rag:sem:"
    semantic_threshold: float = float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.95"))
    semantic_max_entries: int = int(os.getenv("SEMANTIC_CACHE_MAX", "10000"))
 
    # L3 — Embedding cache: chunk text hash → vector (avoids re-embedding)
    embed_ttl: int = int(os.getenv("EMBED_CACHE_TTL", "604800"))        # 7 days
    embed_prefix: str = "rag:emb:"
 
    enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
 
 
# ── Stage 7: Query Optimization ──────────────────────────────

@dataclass
class QueryOptimizerConfig:
    # LLM backend (Groq API)
    groq_api_key: str | None = os.getenv("GROQ_API_KEY", None)
    llm_model: str  = os.getenv("OLLAMA_MODEL", "llama-3.1-8b-instant")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))   # seconds

    # T1 — Query rewriting
    rewrite_enabled: bool = os.getenv("REWRITE_ENABLED", "true").lower() == "true"

    # T2 — HyDE (Hypothetical Document Embeddings)
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    hyde_docs: int     = int(os.getenv("HYDE_DOCS", "3"))   # docs to generate and mean-pool

    # T3 — Multi-query expansion
    multiquery_enabled: bool = os.getenv("MULTIQUERY_ENABLED", "true").lower() == "true"
    multiquery_count: int    = int(os.getenv("MULTIQUERY_COUNT", "4"))  # paraphrases to generate

    # T4 — Sub-question decomposition
    decompose_enabled: bool         = os.getenv("DECOMPOSE_ENABLED", "true").lower() == "true"
    decompose_max_subquestions: int = int(os.getenv("DECOMPOSE_MAX",       "4"))
    decompose_min_words: int        = int(os.getenv("DECOMPOSE_MIN_WORDS", "8"))

    # T5 — Metadata filter extraction (always on, rule-based + LLM)

# ── Stage 8: Re-ranker ────────────────────────────────────────

@dataclass
class RerankerConfig:
    # Backend: "local" (cross-encoder) | "voyage" (API, no RAM cost)
    backend: str     = os.getenv("RERANKER_BACKEND", "local")
    model_name: str  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    device: str      = os.getenv("RERANKER_DEVICE", "cpu")
    top_n: int       = int(os.getenv("RERANKER_TOP_N", "5"))
    batch_size: int  = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
    enabled: bool    = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    score_threshold: float = float(os.getenv("RERANKER_THRESHOLD", "-5.0"))
    # Voyage reranking settings
    voyage_rerank_model: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")


@dataclass
class GeneratorConfig:
    # Groq API (reuses same key as query optimizer)
    groq_api_key: str | None = os.getenv("GROQ_API_KEY", None)
    model: str    = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")
    timeout: int  = int(os.getenv("GENERATOR_TIMEOUT", "60"))
 
    # Generation parameters
    max_tokens: int    = int(os.getenv("GENERATOR_MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.2"))  # Low = factual
    stream: bool       = os.getenv("GENERATOR_STREAM", "true").lower() == "true"
 
    # Context assembly
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))   # Top N chunks to inject
    max_chunk_tokens: int   = int(os.getenv("MAX_CHUNK_TOKENS", "400"))   # Truncate each chunk
 
    # Grounding
    # If best chunk score is below this, refuse to answer (no hallucination)
    min_confidence_score: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "-3.0"))
 
 

# ── Singletons — import these across all modules ──────────────
 
TRANSCRIPT      = TranscriptConfig()
CHUNKING        = ChunkingConfig()
EMBEDDING       = EmbeddingConfig()
QDRANT          = QdrantConfig()
COLLECTION      = CollectionConfig()
BM25            = BM25Config()
FUSION          = FusionConfig()
CACHE           = CacheConfig()
QUERY_OPTIMIZER = QueryOptimizerConfig()
RERANKER = RerankerConfig()
GENERATOR = GeneratorConfig()