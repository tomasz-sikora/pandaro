"""Export / import: the portable ``.pandaro`` bundle plus SRT/VTT/Markdown.

The bundle is a JSON document (optionally zipped by the SPA) carrying the entire
:class:`Analysis` — transcript, diarization, analyses, summary, RAG chunks +
vectors, preset and model versions — so a session can be reloaded *without*
re-running the pipeline.
"""

from __future__ import annotations

import json

from ..schemas import Analysis, Transcript


def _ts_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def transcript_to_srt(transcript: Transcript) -> str:
    lines: list[str] = []
    for i, seg in enumerate(transcript.segments, 1):
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}")
        lines.append(f"{speaker}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def transcript_to_vtt(transcript: Transcript) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in transcript.segments:
        speaker = f"<v {seg.speaker}>" if seg.speaker else ""
        lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
        lines.append(f"{speaker}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def analysis_to_markdown(analysis: Analysis) -> str:
    a = analysis
    out: list[str] = [f"# Raport — {a.media_filename or 'nagranie'}", ""]
    out.append(f"*Czas trwania:* {a.media_duration:.0f} s  ")
    out.append(f"*Pewność rozpoznania:* {a.confidence.mean_word_confidence:.0%}  ")
    out.append("")
    if a.summary.overall:
        out += ["## Podsumowanie", "", a.summary.overall, ""]
    if a.speakers:
        out += ["## Rozmówcy", ""]
        for sp in a.speakers:
            name = sp.name or sp.speaker
            out.append(
                f"- **{name}** — {sp.total_speech_s:.0f}s, "
                f"{sp.gender or '?'}, wiek ~{sp.age or '?'}, "
                f"emocja: {sp.dominant_emotion or '?'}"
            )
        out.append("")
    if a.keywords:
        out += ["## Słowa kluczowe", "", ", ".join(k.term for k in a.keywords[:25]), ""]
    if a.entities:
        out += ["## Encje", ""]
        for e in a.entities[:40]:
            out.append(f"- {e.text} ({e.type})")
        out.append("")
    out += ["## Transkrypt", ""]
    for seg in a.transcript.segments:
        speaker = f"**{seg.speaker}**: " if seg.speaker else ""
        out.append(f"- `{seg.start:.1f}s` {speaker}{seg.text}")
    return "\n".join(out)


def export_bundle(analysis: Analysis) -> str:
    """Serialize the full analysis to a JSON ``.pandaro`` string."""
    return analysis.model_dump_json(indent=2)


def import_bundle(data: str | bytes) -> Analysis:
    """Restore an :class:`Analysis` from a ``.pandaro`` JSON string/bytes."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return Analysis.model_validate(json.loads(data))
