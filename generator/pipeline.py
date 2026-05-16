"""
Stage 9: LLM Generation Pipeline
==================================
Orchestrates the full answer generation step:
  1. Build context from re-ranked chunks (parent_text + citations)
  2. Construct system prompt (grounding rules + context)
  3. Call Groq LLM (streaming or blocking)
  4. Return GenerationResult with answer + citations

This is the final stage before the answer reaches the user.

Full pipeline recap:
  URL → transcript → clean → chunk → embed → Qdrant+BM25
       → cache → query optimize → hybrid retrieve → rerank → generate ← here
"""

from dataclasses import dataclass, field
from typing import Generator as GenType

from generator.context import build_context, BuiltContext, SourceCitation
from generator.prompt import build_prompts
from generator.llm import generate
from reranker.model import RankedChunk
from config import GENERATOR
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger("generator.pipeline")


@dataclass
class GenerationResult:
    """
    Complete output of Stage 9.
    Returned to the caller (main.py or FastAPI endpoint).
    """
    query: str
    answer: str                          # Full generated answer
    citations: list[SourceCitation]      # Source metadata for response footer
    is_confident: bool                   # False = low relevance, hedged answer
    context_chunks_used: int             # How many chunks were injected
    model: str = GENERATOR.model

    def format_citations(self) -> str:
        """Format citation footer for CLI display."""
        if not self.citations:
            return ""
        lines = ["\n── Sources ──────────────────────────────────────────"]
        for c in self.citations:
            lines.append(
                f"[Source {c.index}] {c.video_title} | {c.channel}\n"
                f"           {c.timestamp_label} → {c.url_with_timestamp}"
            )
        return "\n".join(lines)


class Generator:
    """
    Stage 9: generates a grounded, cited answer from re-ranked chunks.

    Usage (blocking):
        gen = Generator()
        result = gen.run(query="...", ranked_chunks=[...])
        print(result.answer)
        print(result.format_citations())

    Usage (streaming):
        for token in gen.stream(query="...", ranked_chunks=[...]):
            print(token, end="", flush=True)
    """

    @timed("stage9.generator")
    def run(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> GenerationResult:
        """
        Generate a blocking (non-streaming) answer.

        Steps:
          1. Build context from chunks
          2. Build system + user prompts
          3. Call LLM (blocking)
          4. Return GenerationResult

        Args:
            query:         Original user question
            ranked_chunks: Output of Stage 8 reranker

        Returns:
            GenerationResult with full answer and citations
        """
        context = build_context(ranked_chunks)
        system_prompt, user_message = build_prompts(query, context)

        answer = generate(system_prompt, user_message, stream=False)

        return GenerationResult(
            query=query,
            answer=answer,
            citations=context.citations,
            is_confident=context.is_confident,
            context_chunks_used=context.chunk_count,
        )

    def stream(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> tuple[GenType[str, None, None], "GenerationResult"]:
        """
        Generate a streaming answer.

        Returns a tuple of:
          - token_generator: yields string chunks as they arrive from Groq
          - result_stub:     GenerationResult with citations (answer="" until complete)

        Usage:
            token_gen, result = gen.stream(query, ranked_chunks)
            full_answer = ""
            for token in token_gen:
                print(token, end="", flush=True)
                full_answer += token
            result.answer = full_answer   # Populate after streaming
            print(result.format_citations())
        """
        context = build_context(ranked_chunks)
        system_prompt, user_message = build_prompts(query, context)

        token_generator = generate(system_prompt, user_message, stream=True)

        result_stub = GenerationResult(
            query=query,
            answer="",   # Populated by caller after consuming the generator
            citations=context.citations,
            is_confident=context.is_confident,
            context_chunks_used=context.chunk_count,
        )

        return token_generator, result_stub