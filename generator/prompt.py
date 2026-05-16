"""
Prompt Builder
===============
Constructs the system and user prompts for the LLM generation step.

Design principles:
  1. Strict grounding — LLM must only answer from provided context
  2. Citation requirement — every claim must reference a [Source N]
  3. Timestamp linking — encourage linking to exact video moments
  4. Graceful refusal — if context is insufficient, say so clearly
  5. Concise answers — no padding, no repetition of the question

System prompt structure:
  - Role definition
  - Grounding rules (answer only from context)
  - Citation format instructions
  - Refusal instructions for out-of-scope questions
"""

from generator.context import BuiltContext


_SYSTEM_PROMPT = """You are a helpful YouTube video knowledge assistant.
You answer questions strictly based on the video transcript excerpts provided below.

## Rules
1. Answer ONLY using information from the provided [Source N] excerpts.
2. Every factual claim must cite its source using [Source N] inline.
3. If multiple sources support a claim, cite all of them: [Source 1][Source 2].
4. If the answer is not found in the sources, say exactly:
   "I couldn't find information about this in the indexed videos."
   Do NOT guess, infer, or use outside knowledge.
5. Be concise. Do not repeat the question. Do not pad with filler phrases.
6. When relevant, mention the video timestamp so the user can watch that moment.

## Video Context
{context}

## Instructions
Answer the user's question using only the context above.
Cite every claim with [Source N]. Be direct and concise."""


_LOW_CONFIDENCE_PROMPT = """You are a helpful YouTube video knowledge assistant.

The search returned results but with low relevance scores, meaning the indexed
videos may not contain a good answer to this question.

Respond with:
"I couldn't find reliable information about this in the indexed videos.
The closest content I found was: [brief 1-sentence description of what was found].
You may want to try rephrasing your question or indexing more relevant videos."

## Closest content found (low confidence)
{context}"""


def build_prompts(
    query: str,
    context: BuiltContext,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for the LLM.

    Args:
        query:   The user's original question
        context: Assembled context from context builder

    Returns:
        Tuple of (system_prompt, user_message)
    """
    if not context.is_confident or context.chunk_count == 0:
        system = _LOW_CONFIDENCE_PROMPT.format(
            context=context.context_text
        )
    else:
        system = _SYSTEM_PROMPT.format(
            context=context.context_text
        )

    return system, query