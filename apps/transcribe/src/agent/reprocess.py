"""
Fragment re-processing — re-run transcription, diarization, or translation on a
selected time range of an already-transcribed recording.

The frontend keeps the source audio and the current segments. To re-process a
fragment it POSTs the audio + the current segments + a [start, end] range + a
mode. We build a lightweight AgentContext, run only the relevant step on the
selected range, splice the result back into the full transcript and stream a
fresh `result` event.

This deliberately does NOT run the full agent loop — re-processing is a focused,
deterministic operation, not an open-ended analysis.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np

from .context import (
    AgentContext, register_session, deregister_session,
)
from .helpers import _send, _progress
from .result_builders import _build_segments_out, _build_speaker_profiles
from .tools import (
    _tool_get_audio_info,
    _tool_retranscribe_time_range,
    _tool_retranslate_segments,
)

logger = logging.getLogger(__name__)

VALID_MODES = ("transcription", "diarization", "translation")


def _seg_mid(seg: Dict) -> float:
    return ((seg.get("start") or 0.0) + (seg.get("end") or 0.0)) / 2.0


def _seg_in_range(seg: Dict, start: float, end: float) -> bool:
    """A segment belongs to the range if its midpoint falls inside it."""
    return start <= _seg_mid(seg) < end


def _mfcc_features(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        import librosa
    except ImportError:
        return None
    if len(audio) < sr * 1.0:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=40)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        return np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(d1, axis=1), np.std(d1, axis=1),
            np.mean(d2, axis=1), np.std(d2, axis=1),
        ])
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _reassign_speakers_by_overlap(new_segs: List[Dict], original: List[Dict]) -> None:
    """Assign each new segment the speaker of the original segment it most
    overlaps with in time. Keeps speaker continuity after re-transcription."""
    for ns in new_segs:
        ns_start, ns_end = ns.get("start") or 0.0, ns.get("end") or 0.0
        best_sp, best_ov = None, 0.0
        for os_ in original:
            ov = min(ns_end, os_.get("end") or 0.0) - max(ns_start, os_.get("start") or 0.0)
            if ov > best_ov:
                best_ov, best_sp = ov, os_.get("speaker")
        if best_sp:
            ns["speaker"] = best_sp


def _remap_fragment_speakers(
    ctx: AgentContext, in_range: List[Dict], frag_segments: List[Dict],
    start: float,
) -> None:
    """Map fragment-local pyannote labels onto the parent's existing speakers
    via MFCC cosine similarity, so re-diarization doesn't invent new labels for
    voices already present in the recording."""
    sr = ctx.sample_rate
    audio = ctx.audio_pcm

    # Build MFCC profile for every existing parent speaker (from full audio).
    parent_audio: Dict[str, List[np.ndarray]] = {}
    for seg in ctx.segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        a = audio[int((seg.get("start") or 0) * sr):int((seg.get("end") or 0) * sr)]
        if len(a):
            parent_audio.setdefault(sp, []).append(a)
    parent_feat = {
        sp: f for sp, chunks in parent_audio.items()
        if (f := _mfcc_features(np.concatenate(chunks)[: sr * 90], sr)) is not None
    }

    # Build MFCC profile for each fragment-local label.
    frag_audio: Dict[str, List[np.ndarray]] = {}
    for seg in frag_segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        a = audio[int((seg.get("start") or 0) * sr):int((seg.get("end") or 0) * sr)]
        if len(a):
            frag_audio.setdefault(sp, []).append(a)

    existing_ids = sorted(parent_feat.keys())
    next_idx = 1 + max(
        [int(s.split("_")[-1]) for s in existing_ids if s.split("_")[-1].isdigit()] or [0]
    )
    mapping: Dict[str, str] = {}
    for frag_sp, chunks in frag_audio.items():
        ffeat = _mfcc_features(np.concatenate(chunks)[: sr * 90], sr)
        best_sp, best_sim = None, 0.0
        if ffeat is not None:
            for psp, pfeat in parent_feat.items():
                sim = _cosine(ffeat, pfeat)
                if sim > best_sim:
                    best_sim, best_sp = sim, psp
        if best_sp is not None and best_sim >= 0.75:
            mapping[frag_sp] = best_sp
        else:
            mapping[frag_sp] = f"GŁOS_{next_idx:02d}"
            next_idx += 1

    for seg in frag_segments:
        seg["speaker"] = mapping.get(seg.get("speaker"), seg.get("speaker"))


def _splice(ctx: AgentContext, start: float, end: float, new_segs: List[Dict]) -> None:
    """Replace segments whose midpoint is inside [start, end] with new_segs."""
    kept = [s for s in ctx.segments if not _seg_in_range(s, start, end)]
    kept.extend(new_segs)
    kept.sort(key=lambda s: s.get("start") or 0.0)
    for i, s in enumerate(kept):
        s["id"] = i
    ctx.segments = kept


def run_reprocess(
    ctx: AgentContext, models: Dict,
    start_sec: float, end_sec: float, mode: str,
) -> None:
    """Re-process a single fragment. Runs in the GPU executor thread."""
    register_session(ctx)
    try:
        _run_reprocess_inner(ctx, models, start_sec, end_sec, mode)
    except Exception as exc:
        logger.exception("Reprocess error: %s", exc)
        _send(ctx, {"type": "error", "message": str(exc)})
    finally:
        deregister_session(ctx.session_id)
        asyncio.run_coroutine_threadsafe(ctx.queue.put(None), ctx.loop)


def _run_reprocess_inner(
    ctx: AgentContext, models: Dict,
    start: float, end: float, mode: str,
) -> None:
    if mode not in VALID_MODES:
        _send(ctx, {"type": "error", "message": f"Nieznany tryb: {mode}"})
        return
    if not ctx.segments:
        _send(ctx, {"type": "error", "message": "Brak segmentów do ponownego przetworzenia."})
        return

    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    in_range = [s for s in ctx.segments if _seg_in_range(s, start, end)]
    if not in_range:
        _send(ctx, {"type": "error", "message": "Wybrany zakres nie zawiera żadnych segmentów."})
        return

    label = {"transcription": "transkrypcji", "diarization": "diaryzacji",
             "translation": "tłumaczenia"}[mode]
    _progress(ctx, "reprocessing", 10,
              f"Ponowne przetwarzanie {label}: {start:.0f}–{end:.0f}s…")

    if mode == "translation":
        ids = [s.get("id") for s in in_range if s.get("id") is not None]
        ctx.do_translate = True
        _tool_retranslate_segments({"segment_ids": ids}, ctx, models)

    elif mode == "transcription":
        original = list(ctx.segments)
        res = _tool_retranscribe_time_range(
            {"start_sec": start, "end_sec": end, "language": ctx.language_hint},
            ctx, models,
        )
        if res.get("error"):
            _send(ctx, {"type": "error", "message": res["error"]})
            return
        # _tool_retranscribe_time_range already spliced; restore speakers via overlap
        new_range = [s for s in ctx.segments if _seg_in_range(s, start, end)]
        _reassign_speakers_by_overlap(new_range, original)
        for i, s in enumerate(ctx.segments):
            s["id"] = i

    else:  # diarization
        _reprocess_diarization(ctx, models, start, end, in_range)

    _emit_result(ctx)


def _reprocess_diarization(
    ctx: AgentContext, models: Dict,
    start: float, end: float, in_range: List[Dict],
) -> None:
    diarizer = models.get("diarizer")
    if diarizer is None or getattr(diarizer, "_method", "none") == "none":
        _send(ctx, {"type": "error", "message": "Diaryzator niedostępny."})
        return

    sr = ctx.sample_rate
    frag = ctx.audio_pcm[int(start * sr):int(end * sr)]
    # Re-diarize the in-range segments using only the fragment audio.
    shifted = [
        {**s, "start": (s.get("start") or 0) - start, "end": (s.get("end") or 0) - start}
        for s in in_range
    ]
    rediar = diarizer.diarize(frag, sr, shifted)
    # Shift timestamps back to absolute.
    for s in rediar:
        s["start"] = round((s.get("start") or 0) + start, 3)
        s["end"] = round((s.get("end") or 0) + start, 3)

    _remap_fragment_speakers(ctx, in_range, rediar, start)
    _splice(ctx, start, end, rediar)

    unique = len({s.get("speaker") for s in ctx.segments})
    _send(ctx, {
        "type": "diarization_update",
        "segments": _build_segments_out(ctx),
        "speaker_profiles": _build_speaker_profiles(ctx),
        "speaker_count": unique,
        "message": f"Ponowna diaryzacja zakończona: {unique} mówca(ów)",
    })


def _emit_result(ctx: AgentContext) -> None:
    _progress(ctx, "done", 100, "Ponowne przetwarzanie zakończone.")
    _send(ctx, {
        "type": "result",
        "segments": _build_segments_out(ctx),
        "detected_language": ctx.detected_language,
        "duration": round(ctx.duration, 2),
        "speaker_profiles": _build_speaker_profiles(ctx),
        "entities": ctx.entities,
        "summary": ctx.summary,
        "report": ctx.report,
        "rag_entries": ctx.rag_entries,
        "quality_stats": ctx.quality_stats,
        "segment_quality": ctx.segment_quality,
        "topics": ctx.topics,
        "asr_engine": ctx.asr_engine,
        "model_used": ctx.ollama_model,
        "reprocessed": True,
    })
