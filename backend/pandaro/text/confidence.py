"""Confidence aggregation for ASR output (section 3 of the plan).

Whisper exposes several signals; we fold them into a single 0..1 per-word /
per-segment confidence and a session-level :class:`ConfidenceReport`.
"""

from __future__ import annotations

import math

from ..schemas import ConfidenceReport, Segment, Transcript


def logprob_to_prob(logprob: float | None) -> float:
    """Map an average log-probability to a 0..1 score."""
    if logprob is None:
        return 1.0
    return max(0.0, min(1.0, math.exp(logprob)))


def word_confidence(prob: float | None, no_speech_prob: float | None = None) -> float:
    """Combine a word probability with the segment's no-speech probability."""
    p = 1.0 if prob is None else max(0.0, min(1.0, prob))
    if no_speech_prob is not None:
        p *= 1.0 - max(0.0, min(1.0, no_speech_prob))
    return p


def is_hallucination(segment: Segment) -> bool:
    """Whisper's classic silence-repetition hallucination heuristic."""
    nsp = segment.no_speech_prob or 0.0
    cr = segment.compression_ratio or 0.0
    return nsp > 0.6 and cr > 2.4


def annotate_transcript(transcript: Transcript, threshold: float) -> ConfidenceReport:
    """Fill in per-word/segment confidence + low-confidence flags, in place.

    Returns a session-level confidence report.
    """
    all_word_conf: list[float] = []
    low_count = 0
    per_speaker_conf: dict[str, list[float]] = {}

    for seg in transcript.segments:
        seg_word_conf: list[float] = []
        for w in seg.words:
            c = word_confidence(w.confidence, seg.no_speech_prob)
            w.confidence = round(c, 4)
            w.low_confidence = c < threshold
            if w.low_confidence:
                low_count += 1
            seg_word_conf.append(c)
            all_word_conf.append(c)
            if seg.speaker:
                per_speaker_conf.setdefault(seg.speaker, []).append(c)

        if seg_word_conf:
            seg.confidence = round(sum(seg_word_conf) / len(seg_word_conf), 4)
        else:
            seg.confidence = round(logprob_to_prob(seg.avg_logprob), 4)

    total = len(all_word_conf)
    mean = sum(all_word_conf) / total if total else 1.0
    return ConfidenceReport(
        mean_word_confidence=round(mean, 4),
        low_confidence_ratio=round(low_count / total, 4) if total else 0.0,
        per_speaker={
            spk: round(sum(v) / len(v), 4) for spk, v in per_speaker_conf.items() if v
        },
    )
