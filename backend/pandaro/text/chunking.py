"""Chunking utilities for RAG indexing and map-reduce summarization.

RAG chunking is *speaker-turn aware*: we never split mid-utterance, we group
consecutive segments of the same speaker up to a token budget with overlap, and
we carry timestamps/speaker/confidence so any retrieved quote can jump to audio.
"""

from __future__ import annotations

from ..schemas import RagChunk, Segment, Transcript
from .phonetic import phonetic_text
from .translit import normalize_text


def _approx_tokens(text: str) -> int:
    # Cheap, language-agnostic token estimate (~4 chars/token).
    return max(1, len(text) // 4)


def chunk_transcript(
    transcript: Transcript,
    *,
    max_tokens: int = 320,
    overlap_segments: int = 1,
) -> list[RagChunk]:
    """Group segments into speaker-turn-aware chunks for the RAG index."""
    chunks: list[RagChunk] = []
    buf: list[Segment] = []
    buf_tokens = 0
    cid = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, cid
        if not buf:
            return
        text = " ".join(s.text.strip() for s in buf).strip()
        translation_parts = [s.translation for s in buf if s.translation]
        translation = " ".join(translation_parts) if translation_parts else None
        confs = [s.confidence for s in buf]
        chunks.append(
            RagChunk(
                id=cid,
                text=text,
                translation=translation,
                speaker=buf[0].speaker,
                start=buf[0].start,
                end=buf[-1].end,
                confidence=round(sum(confs) / len(confs), 4) if confs else 1.0,
                normalized=normalize_text(text),
                phonetic=phonetic_text(text),
            )
        )
        cid += 1
        # Keep an overlap tail for context continuity.
        tail = buf[-overlap_segments:] if overlap_segments else []
        buf = list(tail)
        buf_tokens = sum(_approx_tokens(s.text) for s in buf)

    prev_speaker: str | None = None
    for seg in transcript.segments:
        seg_tokens = _approx_tokens(seg.text)
        speaker_changed = prev_speaker is not None and seg.speaker != prev_speaker
        if buf and (buf_tokens + seg_tokens > max_tokens or speaker_changed):
            flush()
        buf.append(seg)
        buf_tokens += seg_tokens
        prev_speaker = seg.speaker
    flush()
    return chunks


def plan_summary_chunks(text: str, max_chars: int) -> list[str]:
    """Split a long transcript into char-bounded pieces for map-reduce summary.

    Splits on paragraph/line boundaries first to avoid cutting sentences.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        ln = len(line) + 1
        if size + ln > max_chars and current:
            pieces.append("\n".join(current))
            current, size = [], 0
        # A single oversized line is hard-split.
        if ln > max_chars:
            for i in range(0, len(line), max_chars):
                pieces.append(line[i : i + max_chars])
            continue
        current.append(line)
        size += ln
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]
