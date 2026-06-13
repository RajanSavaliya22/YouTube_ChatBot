# YouTube RAG Chatbot — Stages 1–5

Self-hosted YouTube RAG pipeline covering transcript ingestion through
hybrid vector retrieval. Built for production: no external APIs required.

## Architecture

```
YouTube URL
    │
    ▼
Stage 1 ── Transcript        yt-dlp (native captions) → faster-whisper (fallback)
    │
    ▼
Stage 2 ── Cleaning          Filler removal, dedup, merge, punctuation restoration
    │
    ▼
Stage 3 ── Chunking          Hierarchical child/parent chunks with timestamps
    │
    ▼
Stage 4 ── Embeddings        BAAI/bge-large-en-v1.5 (1024-dim, L2-normalized)
    │
    ▼
Stage 5 ── Vector Store
            ├── Qdrant        Dense ANN (HNSW + INT8 quantization)
            ├── BM25          Sparse keyword index (in-memory + pickle)
            └── RRF Fusion    Reciprocal Rank Fusion (k=60)
```

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local Docker setup)
```

### 4. Index a YouTube video

```bash
python main.py index https://www.youtube.com/watch?v=VIDEO_ID
```

Options:
- `--reindex`  Force re-indexing even if video already exists
- `--whisper`  Force Whisper transcription (skip native captions)

### 5. Query

```bash
python main.py query "What did they say about transformers?"

# Filter by channel
python main.py query "explain attention mechanism" --channel="Andrej Karpathy"

# Filter by specific video
python main.py query "gradient descent" --video=VIDEO_ID

# More results
python main.py query "neural networks" --top_k=20
```

### 6. Check stats

```bash
python main.py stats
```

## Project Structure

```
youtube-rag/
├── config.py                       All settings (reads from .env)
├── schema.py                       Shared data models
├── main.py                         CLI: index + query + stats
├── docker-compose.yml              Qdrant service
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── 01_transcript.py            Stage 1: yt-dlp / Whisper
│   ├── 02_cleaner.py               Stage 2: Noise removal + punctuation
│   ├── 03_chunker.py               Stage 3: Hierarchical chunking
│   └── 04_embedder.py              Stage 4: BGE embeddings
│
├── vector_store/
│   ├── client.py                   Qdrant connection + collection setup
│   ├── indexer.py                  Upsert to Qdrant + BM25
│   ├── retriever.py                Hybrid search entry point
│   ├── fusion.py                   Reciprocal Rank Fusion
│   └── sparse/
│       ├── tokenizer.py            Lowercase, stem, stopwords
│       ├── bm25_index.py           BM25 scoring engine
│       └── bm25_store.py           Persistence + payload lookup
│
├── utils/
│   ├── embedder.py                 embed_texts() + embed_query()
│   ├── logger.py                   Structured logging
│   └── timer.py                    Stage timing decorator
│
├── storage/                        Runtime data (gitignored)
│   ├── transcripts/                Raw transcripts per video
│   ├── cleaned/                    Cleaned transcripts
│   ├── chunks/                     Serialized chunk payloads
│   ├── embeddings/                 Cached embedding vectors
│   ├── bm25_index/                 Persisted BM25 pickle
│   └── qdrant_data/                Qdrant volume
│
└── tests/
    ├── test_chunker.py
    ├── test_bm25.py
    └── test_fusion.py
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Note: `test_chunker.py` and `test_bm25.py` and `test_fusion.py` are fully
in-memory and do not require Qdrant to be running.

## Hardware Requirements

| Component        | Minimum              | Recommended          |
|------------------|----------------------|----------------------|
| RAM              | 4 GB                 | 16 GB                |
| Storage          | 10 GB                | 50 GB                |
| GPU              | Not required (CPU)   | Any CUDA GPU         |
| Embedding speed  | ~100 chunks/min CPU  | ~2000 chunks/min GPU |
| Whisper model    | `base` (CPU)         | `medium` (GPU)       |

## Next Stages (not yet implemented)

- **Stage 6**: Re-ranker (`BAAI/bge-reranker-base`)
- **Stage 7**: Query optimization (HyDE, multi-query expansion, sub-questions)
- **Stage 8**: LLM generation with timestamp citations (Llama 3.1 via vLLM)
- **Stage 9**: Redis caching layer
- **Stage 10**: FastAPI REST service + streaming
