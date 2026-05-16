"""
Generator LLM Client
=====================
Calls Groq API for answer generation with optional token streaming.

Streaming:
  When stream=True (default), tokens are yielded one by one as they
  arrive — the caller (main.py / FastAPI) can push them to the user
  in real time, giving a ChatGPT-like experience.

  When stream=False, the full response is returned at once.

Model choice for generation vs query optimization:
  Query optimizer (Stage 7): small fast model (llama-3.1-8b-instant)
    — many calls, short outputs, speed matters
  Generator (Stage 9):      large capable model (llama-3.3-70b-versatile)
    — one call per query, long output, quality matters

Separate config (GENERATOR_MODEL) so you can tune independently.
"""

import os
from typing import Generator as GenType
from config import GENERATOR
from utils.logger import get_logger

logger = get_logger("generator.llm")


def generate(
    system_prompt: str,
    user_message: str,
    stream: bool | None = None,
) -> str | GenType[str, None, None]:
    """
    Generate an answer using Groq.

    Args:
        system_prompt: Built by prompt.py — includes context + grounding rules
        user_message:  The user's raw question
        stream:        Override config stream setting (None = use config)

    Returns:
        If stream=False: full answer string
        If stream=True:  generator yielding text chunks
    """
    use_stream = stream if stream is not None else GENERATOR.stream

    api_key = GENERATOR.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    logger.info(
        f"Generating answer | model={GENERATOR.model} "
        f"stream={use_stream} max_tokens={GENERATOR.max_tokens}"
    )

    if use_stream:
        return _stream(client, messages)
    else:
        return _complete(client, messages)


def _complete(client, messages: list[dict]) -> str:
    """Non-streaming: return full answer at once."""
    response = client.chat.completions.create(
        model=GENERATOR.model,
        messages=messages,
        temperature=GENERATOR.temperature,
        max_tokens=GENERATOR.max_tokens,
        stream=False,
    )
    answer = response.choices[0].message.content.strip()
    logger.info(f"Generated {len(answer.split())} words.")
    return answer


def _stream(client, messages: list[dict]) -> GenType[str, None, None]:
    """
    Streaming: yield text chunks as they arrive from Groq.
    Each yielded value is a partial string (one or more tokens).
    Caller iterates and prints/pushes each chunk.
    """
    response = client.chat.completions.create(
        model=GENERATOR.model,
        messages=messages,
        temperature=GENERATOR.temperature,
        max_tokens=GENERATOR.max_tokens,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content