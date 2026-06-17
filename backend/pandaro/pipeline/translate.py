"""Translation phase: per-segment translation into the preset target (Polish).

Default backend is Gemma via Ollama (good multilingual quality, already running
on the host). Only segments whose detected language differs from the target are
translated; the original text is always preserved.
"""

from __future__ import annotations

from ..clients import OllamaClient
from ..logging_setup import get_logger
from ..schemas import Transcript

log = get_logger("translate")

_LANG_NAMES = {"pl": "polski", "uk": "ukraiński", "ru": "rosyjski", "en": "angielski"}


async def translate_transcript(
    transcript: Transcript,
    *,
    target: str,
    client: OllamaClient,
    source_languages: list[str] | None = None,
) -> Transcript:
    """Translate non-target segments into ``target`` using the LLM, in place."""
    target_name = _LANG_NAMES.get(target, target)
    for seg in transcript.segments:
        seg_lang = seg.language or transcript.language
        if seg_lang == target or not seg.text.strip():
            continue
        if source_languages and seg_lang not in source_languages:
            continue
        prompt = (
            f"Przetłumacz poniższy tekst na język {target_name}. "
            "Zwróć wyłącznie tłumaczenie, bez komentarzy.\n\n"
            f"{seg.text}"
        )
        try:
            seg.translation = await client.generate(prompt, temperature=0.0)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("translate.failed", seg=seg.id, error=str(exc))
    return transcript
