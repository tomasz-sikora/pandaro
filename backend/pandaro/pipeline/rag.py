"""RAG index build phase.

Chunks the transcript (speaker-turn-aware), computes Slavic-aware search aids
(normalized + phonetic columns), and embeds each chunk via Ollama (bge-m3). The
chunks + vectors are shipped to the browser, which builds the actual WASM
SQLite-FTS5 + vector index. The server keeps nothing.
"""

from __future__ import annotations

from ..clients import OllamaClient
from ..logging_setup import get_logger
from ..schemas import RagChunk, Transcript
from ..text import chunk_transcript

log = get_logger("rag")


async def build_rag_chunks(
    transcript: Transcript,
    *,
    client: OllamaClient | None = None,
    embed: bool = True,
    max_tokens: int = 320,
) -> list[RagChunk]:
    chunks: list[RagChunk] = chunk_transcript(transcript, max_tokens=max_tokens)
    if embed and client is not None and chunks:
        texts = [c.text for c in chunks]
        try:
            vectors = await client.embed(texts)
            for c, v in zip(chunks, vectors, strict=False):
                c.embedding = v
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("rag.embed_failed", error=str(exc))
    return chunks
