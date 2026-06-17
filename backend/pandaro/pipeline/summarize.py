"""Summarization phase: hierarchical map-reduce for long (2h+) transcripts.

Even with a 128K-context Gemma, a 2h recording (~20-30k+ words) benefits from
map-reduce: summarize each chunk, then reduce the partial summaries into a final
summary. This bounds prompt size, keeps quality, and supports per-speaker output.
"""

from __future__ import annotations

from ..clients import OllamaClient
from ..config import get_settings
from ..logging_setup import get_logger
from ..schemas import Summary, Transcript
from ..text import plan_summary_chunks

log = get_logger("summarize")

_STYLE = {
    "bullet": "w punktach",
    "narrative": "w formie zwięzłej narracji",
    "minutes": "w formie protokołu ze spotkania (decyzje, ustalenia, zadania)",
}


async def _map(chunk: str, style: str, client: OllamaClient) -> str:
    prompt = (
        f"Streść poniższy fragment rozmowy {style}. "
        "Zachowaj nazwiska, liczby, daty i ustalenia. Odpowiedz po polsku.\n\n"
        f"{chunk}"
    )
    return await client.generate(prompt, temperature=0.2)


async def _reduce(partials: list[str], style: str, client: OllamaClient) -> str:
    joined = "\n\n".join(f"- {p}" for p in partials)
    prompt = (
        f"Na podstawie poniższych streszczeń fragmentów stwórz jedno spójne "
        f"streszczenie całej rozmowy {style}. Odpowiedz po polsku.\n\n{joined}"
    )
    return await client.generate(prompt, temperature=0.2)


async def summarize(
    transcript: Transcript,
    *,
    client: OllamaClient,
    style: str = "bullet",
    per_speaker: bool = True,
) -> Summary:
    settings = get_settings()
    style_text = _STYLE.get(style, _STYLE["bullet"])
    full_text = transcript.text

    chunks = plan_summary_chunks(full_text, settings.summary_chunk_chars)
    if not chunks:
        return Summary()

    partials = [await _map(c, style_text, client) for c in chunks]
    overall = partials[0] if len(partials) == 1 else await _reduce(partials, style_text, client)

    per_speaker_summaries: dict[str, str] = {}
    if per_speaker:
        by_speaker: dict[str, list[str]] = {}
        for seg in transcript.segments:
            if seg.speaker:
                by_speaker.setdefault(seg.speaker, []).append(seg.text)
        for spk, texts in by_speaker.items():
            joined = " ".join(texts)[: settings.summary_chunk_chars]
            try:
                per_speaker_summaries[spk] = await _map(joined, style_text, client)
            except Exception:  # pragma: no cover - network dependent
                pass

    return Summary(overall=overall, per_speaker=per_speaker_summaries, topics=[])
