"""
Technique 4: Sub-Question Decomposition
=========================================
Break complex, multi-part questions into atomic sub-questions,
retrieve for each independently, then merge all results.

When to trigger:
  - Query has >= 8 words (DECOMPOSE_MIN_WORDS)
  - Query contains comparison/contrast words ("vs", "compare", "difference")
  - Query contains temporal words ("2022 vs 2024", "before", "after", "changed")
  - Query contains multiple distinct concepts

Examples where decomposition helps:

  Complex: "Compare what Andrej Karpathy said about LLMs in 2022 vs 2024
            and how his views on scaling changed"
  Sub-questions:
    1. "What did Andrej Karpathy say about LLMs in 2022?"
    2. "What did Andrej Karpathy say about LLMs in 2024?"
    3. "How did Karpathy's views on scaling laws change over time?"

  Complex: "What are the pros and cons of fine-tuning vs RAG for production?"
  Sub-questions:
    1. "Advantages of fine-tuning language models for production"
    2. "Disadvantages of fine-tuning compared to RAG"
    3. "Benefits of RAG for production AI systems"
    4. "When to choose RAG over fine-tuning"

Fallback: returns [original_query] if LLM unavailable or query is simple.
"""

import re
from query_optimizer.llm_client import call_llm
from config import QUERY_OPTIMIZER
from utils.logger import get_logger

logger = get_logger("query_optimizer.decomposer")

# Trigger words that suggest a complex multi-part query
_COMPLEX_TRIGGERS = re.compile(
    r"\b(vs|versus|compare|comparison|difference|between|contrast|"
    r"pros and cons|advantages|disadvantages|before|after|"
    r"changed|evolved|progression|also|additionally|furthermore|"
    r"both|either|neither|relationship|impact|effect on)\b",
    re.IGNORECASE,
)

_SYSTEM = """You are a question decomposition assistant.
Break down complex, multi-part questions into simple, atomic sub-questions.
Each sub-question should be answerable independently.
Return ONLY a numbered list of sub-questions, one per line.
No explanations, no preamble. Maximum {max} sub-questions."""

_PROMPT = """Break this complex question into {max} or fewer simple sub-questions:

Question: {query}

Sub-questions:"""


def _parse_numbered_list(text: str) -> list[str]:
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        cleaned = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
        cleaned = cleaned.strip('"').strip("'")
        if cleaned and len(cleaned) > 5:
            # Ensure it ends with a question mark if it looks like a question
            if cleaned[-1] not in ".?!":
                cleaned += "?"
            questions.append(cleaned)
    return questions


def _is_complex(query: str) -> bool:
    """Heuristic: decide if a query is complex enough to decompose."""
    word_count = len(query.split())
    has_trigger = bool(_COMPLEX_TRIGGERS.search(query))
    return word_count >= QUERY_OPTIMIZER.decompose_min_words and has_trigger


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into atomic sub-questions.

    Args:
        query: User query (potentially complex/multi-part)

    Returns:
        List of sub-questions if decomposed, or [original_query] if:
          - Decomposition is disabled
          - Query is not complex enough
          - LLM is unavailable
    """
    if not QUERY_OPTIMIZER.decompose_enabled:
        return [query]

    if not _is_complex(query):
        logger.debug(f"Decomposition skipped — query not complex enough: '{query[:60]}'")
        return [query]

    response = call_llm(
        _PROMPT.format(query=query, max=QUERY_OPTIMIZER.decompose_max_subquestions),
        system=_SYSTEM.format(max=QUERY_OPTIMIZER.decompose_max_subquestions),
    )

    if not response:
        logger.debug("Decomposition skipped — LLM unavailable.")
        return [query]

    sub_questions = _parse_numbered_list(response)

    if len(sub_questions) < 2:
        logger.debug("Decomposition produced < 2 sub-questions — using original.")
        return [query]

    sub_questions = sub_questions[:QUERY_OPTIMIZER.decompose_max_subquestions]

    logger.info(
        f"Decomposed: '{query[:50]}' → {len(sub_questions)} sub-questions"
    )
    for i, q in enumerate(sub_questions, 1):
        logger.info(f"  {i}. {q}")

    return sub_questions