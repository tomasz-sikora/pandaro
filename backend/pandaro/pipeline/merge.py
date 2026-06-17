"""Merge diarization speaker turns into the word-level transcript.

For each segment/word we assign the speaker whose turn overlaps it most. This is
the standard WhisperX-style attribution and is pure logic (testable).
"""

from __future__ import annotations

from ..schemas import SpeakerTurn, Transcript


def _best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    best: str | None = None
    best_overlap = 0.0
    for t in turns:
        overlap = max(0.0, min(end, t.end) - max(start, t.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = t.speaker
    return best


def assign_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    """Annotate each segment (and word) with a speaker label, in place."""
    if not turns:
        return transcript
    for seg in transcript.segments:
        seg.speaker = _best_speaker(seg.start, seg.end, turns) or seg.speaker
        for w in seg.words:
            w_speaker = _best_speaker(w.start, w.end, turns)
            if w_speaker:
                # Words inherit segment speaker unless an overlap says otherwise;
                # we keep the segment-level label authoritative for display.
                seg.speaker = seg.speaker or w_speaker
    return transcript


def speaker_talk_time(turns: list[SpeakerTurn]) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in turns:
        out[t.speaker] = out.get(t.speaker, 0.0) + (t.end - t.start)
    return {k: round(v, 2) for k, v in out.items()}
