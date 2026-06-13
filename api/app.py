"""
Stage 10: FastAPI Application
==============================
Production REST API wrapping the full YouTube RAG pipeline (stages 1-9).

Endpoints:
  GET  /              — API info
  GET  /health        — liveness / readiness
  GET  /stats         — per-layer stats
  GET  /videos        — list indexed videos
  DELETE /videos/{id} — remove a video
  POST /index         — index a YouTube video (async)
  GET  /index/status/{id} — poll indexing job
  POST /query         — ask a question (blocking JSON)
  POST /query/stream  — ask a question (SSE streaming)

Run locally:
  uvicorn api.app:app --reload --port 8000

Deploy on Render:
  python -m uvicorn api.app:app --host 0.0.0.0 --port $PORT
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.deps import startup, shutdown
from api.routes import health, index, query
from config import API
from utils.logger import get_logger

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="YouTube RAG Chatbot API",
        description=(
            "Self-hosted YouTube RAG pipeline. "
            "Index videos, ask questions, get grounded answers with timestamp citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allows Chrome extension and any frontend to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip large JSON responses
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Routers
    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(query.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "YouTube RAG Chatbot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()