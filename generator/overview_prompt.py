"""
Overview Prompt Builder
========================
Constructs prompts for general/overview queries where the context
is the full (sampled) transcript rather than retrieved chunks.

Unlike the standard prompt (which uses [Source N] citation format),
this prompt:
  - Tells the LLM it has the full video transcript
  - Asks it to synthesize across the whole content
  - Uses [MM:SS] timestamp references instead of [Source N]
  - Has a different tone: analytical summary vs factual Q&A
"""

from generator.transcript_context import TranscriptContext


_OVERVIEW_SYSTEM = """You are an expert video analyst assistant.
You have been given a transcript excerpt from a YouTube video and must answer
the user's question about the video's overall content.

## Video Information
Title:   {video_title}
Channel: {channel}
Sampled: {chunks_used} segments from {total_chunks} total

## Transcript (chronological excerpts with timestamps)
{context}

## Instructions
- Answer based only on the transcript provided above
- Reference specific moments using their [MM:SS] timestamps when relevant
- If the video covers multiple topics, organize your answer clearly
- Be comprehensive but concise — the user wants to understand the video overall
- If the transcript doesn't give enough information to answer fully, say so"""


def build_overview_prompts(
    query: str,
    ctx: TranscriptContext,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for overview queries.

    Args:
        query: User's general question
        ctx:   TranscriptContext with sampled chunks

    Returns:
        Tuple of (system_prompt, user_message)
    """
    system = _OVERVIEW_SYSTEM.format(
        video_title=ctx.video_title,
        channel=ctx.channel,
        chunks_used=ctx.chunks_used,
        total_chunks=ctx.total_chunks_available,
        context=ctx.context_text,
    )
    return system, query