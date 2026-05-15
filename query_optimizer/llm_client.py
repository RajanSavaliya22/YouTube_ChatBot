"""
LLM Client
===========
Thin wrapper around Ollama for query optimization tasks.
All query optimizer techniques (rewrite, HyDE, multi-query, decompose)
call through here so the LLM backend can be swapped in one place.

Self-hosted via Ollama: https://ollama.ai
  ollama pull llama3.1:8b

Graceful fallback: if Ollama is unreachable, returns None so the
optimizer silently skips that technique and uses the raw query.
"""

import json
import urllib.request
import urllib.error
from config import QUERY_OPTIMIZER
from utils.logger import get_logger
import os
from groq import Groq

logger = get_logger("query_optimizer.llm")


# def _call_ollama(prompt: str, system: str = "") -> str | None:
#     """
#     Call Ollama /api/generate endpoint.
#     Returns response text or None on failure.
#     """
#     payload = {
#         "model": QUERY_OPTIMIZER.ollama_model,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": 0.3,     # Low temp for consistent, factual rewrites
#             "num_predict": 512,     # Max tokens to generate
#         },
#     }
#     if system:
#         payload["system"] = system

#     try:
#         data = json.dumps(payload).encode()
#         req = urllib.request.Request(
#             f"{QUERY_OPTIMIZER.ollama_host}/api/generate",
#             data=data,
#             headers={"Content-Type": "application/json"},
#             method="POST",
#         )
#         with urllib.request.urlopen(req, timeout=QUERY_OPTIMIZER.llm_timeout) as resp:
#             result = json.loads(resp.read())
#             return result.get("response", "").strip()

#     except urllib.error.URLError as e:
#         logger.warning(f"Ollama unreachable at {QUERY_OPTIMIZER.ollama_host}: {e}")
#         return None
#     except Exception as e:
#         logger.warning(f"LLM call failed: {e}")
#         return None

def _call_groq(prompt: str, system: str = "") -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set.")
        return None

    try:
        client = Groq(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            messages=messages,
            model=QUERY_OPTIMIZER.llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None

def call_llm(prompt: str, system: str = "") -> str | None:
    """Public entry point. Currently wraps Ollama; swap here for vLLM/OpenAI."""
    return _call_groq(prompt, system)


def is_available() -> bool:
    """Check if the LLM backend is reachable."""
    try:
        req = urllib.request.Request(
            f"{QUERY_OPTIMIZER.ollama_host}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False