from __future__ import annotations
from .context import AgentContext
from .helpers import (
    _send, _progress, _agent_event,
    _call_ollama_generate, _call_ollama_embed,
    _decode_audio_bytes, OLLAMA_URL, OLLAMA_EMBEDDING_MODEL,
    SUMMARY_NUM_CTX, KEEP_ALIVE,
)
from .result_builders import _build_segments_out, _build_speaker_profiles
from ..memory import save_memory as _save_memory_fn
import asyncio, io, json, logging, os, re, time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
import httpx, numpy as np
logger = logging.getLogger(__name__)

def _tool_probe_audio_fragment(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Transcribe 2-3 short fragments at different positions to calibrate parameters.

    Sampling multiple positions avoids basing the entire initial analysis on one
    lucky (or unlucky) excerpt — e.g. the recording may start with silence or
    music. Returns aggregate confidence and per-position language detection.
    """
    if ctx.audio_pcm is None:
        # Auto-decode first
        _tool_get_audio_info({}, ctx, models)

    duration_sec = min(float(args.get("duration_sec") or 60.0), 90.0)
    language = args.get("language") or ctx.language_hint
    total_dur = ctx.duration

    sr = ctx.sample_rate

    # Choose probe positions — spread across the recording for representativeness
    explicit_start = args.get("start_sec")
    if explicit_start is not None:
        probe_positions = [float(explicit_start)]
    elif total_dur > duration_sec * 3:
        probe_positions = [
            max(30.0, total_dur * 0.15),
            total_dur * 0.50,
            total_dur * 0.80,
        ]
    elif total_dur > duration_sec * 1.5:
        probe_positions = [
            max(10.0, total_dur * 0.20),
            total_dur * 0.65,
        ]
    else:
        probe_positions = [max(0.0, min(30.0, total_dur * 0.2))]

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available for probe"}

    models.get("ensure_whisper_gpu", lambda: None)()

    all_word_probs: List[float] = []
    all_texts: List[str] = []
    languages_detected: Dict[str, int] = {}
    fragment_results: List[Dict] = []

    for pos in probe_positions:
        start_smp = int(max(0.0, min(pos, total_dur - duration_sec)) * sr)
        end_smp = min(start_smp + int(duration_sec * sr), len(ctx.audio_pcm))
        fragment = ctx.audio_pcm[start_smp:end_smp]
        frag_label = f"{pos:.0f}–{pos + duration_sec:.0f}s"

        _progress(ctx, "transcribing", 5, f"Próbkowanie {frag_label}…")

        try:
            probe_chunks, detected_lang, _ = whisper.transcribe(
                fragment, language, progress_cb=lambda *a: None,
                extra_kw={"beam_size": 3, "best_of": 1},  # fast probe
            )
        except Exception as exc:
            fragment_results.append({"fragment": frag_label, "error": str(exc)})
            continue

        frag_probs: List[float] = []
        frag_texts: List[str] = []
        for ch in probe_chunks:
            frag_texts.append(ch.get("text", ""))
            for w in (ch.get("words") or []):
                p = float(w.get("probability", 1.0))
                frag_probs.append(p)
                all_word_probs.append(p)

        frag_conf = round(sum(frag_probs) / len(frag_probs), 3) if frag_probs else 0.0
        sample_text = " ".join(frag_texts)[:300]
        all_texts.append(sample_text)
        languages_detected[detected_lang] = languages_detected.get(detected_lang, 0) + 1
        fragment_results.append({
            "fragment": frag_label,
            "detected_language": detected_lang,
            "segment_count": len(probe_chunks),
            "avg_confidence": frag_conf,
            "sample_text": sample_text[:150],
        })

    avg_conf = round(sum(all_word_probs) / len(all_word_probs), 3) if all_word_probs else 0.0
    dominant_lang = max(languages_detected, key=languages_detected.__getitem__) if languages_detected else (language or "auto")

    # Aggregate parameter suggestions
    suggestions: Dict[str, Any] = {}
    if avg_conf < 0.6:
        suggestions["beam_size"] = 7
        suggestions["vad_filter_threshold"] = 0.15
        suggestions["note"] = "Low confidence across samples — increase beam_size, lower VAD threshold"
    elif avg_conf < 0.75:
        suggestions["beam_size"] = 5
        suggestions["note"] = "Medium confidence — default params should work"
    else:
        suggestions["note"] = "Good confidence — default params are fine"

    total_segs = sum(r.get("segment_count", 0) for r in fragment_results if "segment_count" in r)
    if total_segs == 0:
        suggestions["vad_filter_threshold"] = 0.1
        suggestions["note"] = "No segments detected in any sample — significantly lower VAD threshold"

    return {
        "detected_language": dominant_lang,
        "fragments_probed": len(probe_positions),
        "avg_confidence": avg_conf,
        "fragment_results": fragment_results,
        "sample_text": " ".join(all_texts)[:400],
        "suggestions": suggestions,
    }


def _tool_detect_speaker_count(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Estimate speaker count using pyannote.

    Strategy (most accurate first):
    1. For recordings <=15 min: run pyannote on the FULL audio.
       This is the only way to find ALL speakers — fragment sampling misses
       speakers who only appear in one part (e.g. brief callers, guests).
    2. For longer recordings: sample from 4+ evenly-spaced positions and take max.

    The gold-standard approach for pyannote speaker counting is the full audio.
    """
    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    diarizer = models.get("diarizer")
    if diarizer is None or diarizer._method == "none":
        return {"error": "Diarizer (pyannote) not available", "estimated_speakers": 2}

    sr = ctx.sample_rate
    total_dur = ctx.duration

    # For short recordings: run on the FULL audio for complete speaker coverage
    if total_dur <= 15 * 60:
        _progress(ctx, "diarizing", 5,
                  f"Wykrywanie liczby mówców (pełne nagranie {total_dur/60:.1f} min)…")
        try:
            n = diarizer.count_speakers(ctx.audio_pcm, sr)
            logger.info("Full-audio speaker count: %d", n)
            return {
                "estimated_speakers": n,
                "method": "full_audio",
                "duration_analyzed_sec": round(total_dur, 1),
                "recommendation": f"Use num_speakers={n} in diarize_audio",
            }
        except Exception as exc:
            logger.warning("Full-audio speaker count failed: %s — falling back to fragments", exc)

    # For long recordings: sample multiple positions
    duration_sec = min(float(args.get("duration_sec") or 120.0), 180.0)
    n_probes = max(4, int(total_dur / 120))  # one probe per 2 minutes, min 4
    probe_positions = [total_dur * i / n_probes for i in range(n_probes)]

    _progress(ctx, "diarizing", 5,
              f"Wykrywanie liczby mówców ({n_probes} fragmentów po {duration_sec:.0f}s)…")

    counts: List[int] = []
    fragment_labels: List[str] = []
    for pos in probe_positions:
        start_smp = int(max(0.0, min(pos, total_dur - duration_sec)) * sr)
        end_smp = min(start_smp + int(duration_sec * sr), len(ctx.audio_pcm))
        fragment = ctx.audio_pcm[start_smp:end_smp]
        frag_label = f"{pos:.0f}–{pos + duration_sec:.0f}s"
        try:
            n = diarizer.count_speakers(fragment, sr)
            counts.append(n)
            fragment_labels.append(f"{frag_label}:{n}")
        except Exception as exc:
            logger.warning("Speaker count failed at %s: %s", frag_label, exc)

    if not counts:
        return {"estimated_speakers": 2, "error": "All speaker count attempts failed"}

    n_total = max(counts)
    return {
        "estimated_speakers": n_total,
        "method": "fragment_sampling",
        "counts_per_fragment": fragment_labels,
        "fragments_sampled": len(probe_positions),
        "recommendation": f"Use num_speakers={n_total} in diarize_audio",
    }


def _tool_set_transcription_params(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Store tuning parameters to be used by the next transcribe_audio call."""
    whisper_allowed = {
        "vad_filter_threshold", "beam_size", "temperature",
        "no_speech_threshold", "compression_ratio_threshold",
        "chunk_minutes", "overlap_seconds", "adaptive_vad", "ipa_threshold",
    }
    stored = {}
    for k, v in args.items():
        if k in whisper_allowed and v is not None:
            ctx.transcription_params[k] = v
            stored[k] = v
    # Context management params
    if "max_ctx_segments" in args and args["max_ctx_segments"] is not None:
        ctx.max_ctx_segments = int(args["max_ctx_segments"])
        stored["max_ctx_segments"] = ctx.max_ctx_segments
    return {"stored_params": ctx.transcription_params, "applied": stored,
            "max_ctx_segments": ctx.max_ctx_segments}


def _tool_get_audio_info(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Decode audio to PCM and return metadata."""
    import hashlib
    samples, sr = _decode_audio_bytes(ctx.audio_content, ctx.filename)
    ctx.audio_pcm = samples
    ctx.sample_rate = sr
    ctx.duration = len(samples) / sr
    ctx.audio_sha = hashlib.sha256(ctx.audio_content).hexdigest()
    return {
        "duration_seconds": round(ctx.duration, 1),
        "duration_minutes": round(ctx.duration / 60, 1),
        "filename": ctx.filename,
        "language_hint": ctx.language_hint or "auto",
        "translate_requested": ctx.do_translate,
        "sample_rate": sr,
    }


def _tool_transcribe_audio(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Run full ASR. Uses ctx.transcription_params if set. Streams segment_chunk events."""
    if ctx.audio_pcm is None:
        raise RuntimeError("Audio not decoded yet. Call get_audio_info first.")

    engine = args.get("engine", "whisper").lower()
    language = args.get("language") or ctx.language_hint
    ctx.asr_engine = engine

    def prog(pct, msg):
        _progress(ctx, "transcribing", pct, msg)

    # Streaming callback: emit raw segments chunk-by-chunk as Whisper processes audio
    _streaming_offset = [0]  # mutable counter for global segment index

    def on_chunk_segments(chunk_segs: List[Dict], offset_sec: float) -> None:
        if not chunk_segs:
            return
        start_idx = _streaming_offset[0]
        preview = [
            {
                "id": start_idx + i,
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": "—",  # pre-diarization placeholder
                "language": ctx.language_hint or "?",
            }
            for i, s in enumerate(chunk_segs)
        ]
        _streaming_offset[0] += len(chunk_segs)
        _send(ctx, {
            "type": "segment_chunk",
            "segments": preview,
            "offset_sec": offset_sec,
            "cumulative": _streaming_offset[0],
        })

    if engine == "vibevoice":
        transcriber = models["vibevoice_loader"](progress_cb=prog)
    else:
        models.get("ensure_whisper_gpu", lambda: None)()
        transcriber = models.get("whisper")

    _progress(ctx, "transcribing", 15, f"ASR: {engine} (params: {ctx.transcription_params or 'default'})…")

    extra_kw = dict(ctx.transcription_params) if ctx.transcription_params else None
    transcribe_kwargs: Dict[str, Any] = {"progress_cb": prog}
    if extra_kw and engine == "whisper":
        transcribe_kwargs["extra_kw"] = extra_kw
    if engine == "whisper":
        transcribe_kwargs["on_segments_cb"] = on_chunk_segments

    chunks, detected_lang, dur = transcriber.transcribe(ctx.audio_pcm, language, **transcribe_kwargs)
    if not chunks:
        raise ValueError("ASR returned no segments.")

    ctx.segments = chunks
    ctx.detected_language = detected_lang
    if dur:
        ctx.duration = dur

    return {
        "segment_count": len(chunks),
        "detected_language": detected_lang,
        "duration_seconds": round(ctx.duration, 1),
        "engine_used": engine,
        "params_used": ctx.transcription_params or "defaults",
    }


def _tool_verify_transcript_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Analyse transcript quality and emit a quality_report event."""
    if not ctx.segments:
        return {"error": "No segments to verify. Run transcribe_audio first."}

    low_conf: List[Dict] = []
    all_probs: List[float] = []
    repetitions = 0
    very_short = 0

    for i, seg in enumerate(ctx.segments):
        words = seg.get("words") or []
        probs = [float(w.get("probability", 1.0)) for w in words]
        if probs:
            seg_avg = sum(probs) / len(probs)
            all_probs.extend(probs)
            if seg_avg < 0.55:
                low_conf.append({"id": i, "start": seg.get("start"), "text": seg.get("text", "")[:60], "confidence": round(seg_avg, 3)})

        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur < 0.4:
            very_short += 1

        # Repetition check vs prev segment
        if i > 0:
            prev_text = ctx.segments[i - 1].get("text", "").lower().strip()
            curr_text = seg.get("text", "").lower().strip()
            if prev_text and curr_text and prev_text == curr_text:
                repetitions += 1

    avg_conf = round(sum(all_probs) / len(all_probs), 3) if all_probs else 1.0

    # Find long gaps
    gaps: List[Dict] = []
    for i in range(1, len(ctx.segments)):
        gap = (ctx.segments[i].get("start") or 0) - (ctx.segments[i - 1].get("end") or 0)
        if gap > 10.0:
            gaps.append({"before_segment": i, "gap_sec": round(gap, 1)})

    warnings: List[str] = []
    if avg_conf < 0.65:
        warnings.append(f"Low average confidence ({avg_conf:.2f}) — consider re-transcribing with higher beam_size or lower VAD threshold")
    if len(low_conf) > len(ctx.segments) * 0.2:
        warnings.append(f"{len(low_conf)} segments ({len(low_conf)*100//len(ctx.segments)}%) below confidence threshold")
    if repetitions > 3:
        warnings.append(f"{repetitions} consecutive duplicate segments detected — possible hallucinations")
    if len(gaps) > 0:
        warnings.append(f"{len(gaps)} long silence gaps (>10s) — verify audio continuity")

    ctx.quality_stats = {
        "avg_confidence": avg_conf,
        "low_confidence_count": len(low_conf),
        "repetitions": repetitions,
        "very_short_segments": very_short,
        "long_gaps": len(gaps),
        "total_segments": len(ctx.segments),
    }

    # Store per-segment confidence for UI visualization
    for i, seg in enumerate(ctx.segments):
        words = seg.get("words") or []
        probs = [float(w.get("probability", 1.0)) for w in words]
        if probs:
            ctx.segment_quality[i] = round(sum(probs) / len(probs), 3)

    _agent_event(ctx, "quality_report",
                 avg_confidence=avg_conf,
                 low_confidence_segments=low_conf[:10],
                 warnings=warnings,
                 gaps=gaps[:5],
                 stats=ctx.quality_stats)

    return {
        **ctx.quality_stats,
        "warnings": warnings,
        "recommend_retranscribe": avg_conf < 0.6 or repetitions > 5,
    }


def _tool_merge_short_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Merge very short or closely-spaced same-speaker segments."""
    if not ctx.segments:
        return {"skipped": True, "reason": "No segments"}

    # Default gap=0.05s — only merge truly back-to-back same-speaker fragments
    # (e.g. Whisper split one word across two segments). This conservative
    # threshold alone preserves short interjections; no curated word list needed.
    gap_sec = float(args.get("gap_sec") or 0.05)
    before = len(ctx.segments)

    merged: List[Dict] = []
    for seg in ctx.segments:
        if not merged:
            merged.append(dict(seg))
            continue

        prev = merged[-1]
        gap = (seg.get("start") or 0) - (prev.get("end") or 0)
        same_speaker = prev.get("speaker") == seg.get("speaker")

        # Merge only when the same speaker continues with essentially no gap.
        if same_speaker and gap <= gap_sec:
            prev["text"] = (prev.get("text", "") + " " + seg.get("text", "")).strip()
            prev["text_pl"] = (prev.get("text_pl", "") + " " + seg.get("text_pl", "")).strip() if prev.get("text_pl") else None
            prev["end"] = seg.get("end")
            prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
        else:
            merged.append(dict(seg))

    # Re-index
    for i, s in enumerate(merged):
        s["id"] = i

    ctx.segments = merged
    merged_count = before - len(merged)

    # Emit segment_update so UI reflects new segment boundaries
    if merged_count > 0:
        _send(ctx, {
            "type": "segment_update",
            "operation": "merge",
            "segments": _build_segments_out(ctx),
            "before": before,
            "after": len(merged),
            "merged_count": merged_count,
        })

    return {"before": before, "after": len(merged), "merged_count": merged_count}


def _tool_diarize_audio(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    if ctx.asr_engine == "vibevoice":
        return {"skipped": True, "reason": "vibevoice has built-in diarization"}

    diarizer = models.get("diarizer")
    if diarizer is None:
        return {"skipped": True, "reason": "diarizer not loaded"}

    num_speakers = int(args.get("num_speakers") or 0)
    _progress(ctx, "diarizing", 65, "Identyfikacja mówców (pyannote)…")
    ctx.segments = diarizer.diarize(
        ctx.audio_pcm, ctx.sample_rate, ctx.segments,
        num_speakers=num_speakers if num_speakers > 0 else None,
    )
    unique = len({s["speaker"] for s in ctx.segments})

    # ── Emit diarization_update so UI can refresh speaker labels ──────────
    _send(ctx, {
        "type": "diarization_update",
        "segments": _build_segments_out(ctx),
        "speaker_profiles": _build_speaker_profiles(ctx),
        "speaker_count": unique,
        "message": f"Diaryzacja zakończona: {unique} mówca(ów)",
    })

    return {"speaker_count": unique}


def _tool_profile_speakers(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    profiler = models.get("profiler")
    audio_features = models.get("audio_features")
    _progress(ctx, "profiling", 74, "Analiza cech mówców…")
    if profiler:
        ctx.speaker_profiles_raw = profiler.profile_speakers(ctx.audio_pcm, ctx.sample_rate, ctx.segments)
    if audio_features and audio_features.loaded_extractors:
        try:
            ctx.audio_features_raw = audio_features.extract_per_speaker(ctx.audio_pcm, ctx.sample_rate, ctx.segments)
        except Exception as exc:
            logger.warning("Audio feature extraction failed: %s", exc)
    return {"speakers_profiled": list(ctx.speaker_profiles_raw.keys())}


def _tool_translate_to_polish(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Translation sub-agent: batches → stream → LLM quality check → targeted retry.

    The tool operates as a mini-loop:
    1. Translate all batches (streaming translation_chunk events to UI)
    2. Sample-check quality with LLM on every N-th batch
    3. Collect poor segments (score < 3) for a second pass
    4. Re-translate poor segments with an improved prompt
    5. Return quality report
    """
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    if ctx.detected_language == "pl" or not ctx.do_translate:
        for s in ctx.segments:
            s["text_pl"] = s["text"]
        return {"skipped": True, "reason": f"language={ctx.detected_language}, translate={ctx.do_translate}"}

    from ..translator import translate_segments_to_polish

    quality_scores: List[float] = []
    low_quality_ids: List[int] = []
    batches_since_check = [0]

    def on_batch_done(batch_updates: List[Dict]) -> None:
        if not batch_updates:
            return
        _send(ctx, {"type": "translation_chunk", "updates": batch_updates})
        batches_since_check[0] += 1

        # Quality-check every 2nd batch (previously 3rd — catch problems sooner)
        if batches_since_check[0] % 2 != 0 or ctx.detected_language == "auto":
            return

        # Sample up to 4 segments from this batch for quality scoring
        sample = batch_updates[:4]
        if not sample:
            return

        src_lang = ctx.detected_language
        lines = "\n".join(
            f"{i+1}. [{src_lang}] {ctx.segments[u['idx']].get('text','')[:120]}\n"
            f"   [pl]   {u['text_pl'][:120]}"
            for i, u in enumerate(sample)
        )
        qprompt = (
            f"Oceń jakość tłumaczeń z {src_lang} na polski. "
            "Dla każdej pary podaj ocenę 1-5 (1=błąd/inny język, 3=OK, 5=świetne naturalne tłumaczenie).\n\n"
            f"{lines}\n\n"
            "Zwróć JSON: {\"scores\": [int, ...], \"avg\": float, \"issues\": [str, ...]}"
        )
        qschema = {
            "type": "object",
            "properties": {
                "scores": {"type": "array", "items": {"type": "integer"}},
                "avg": {"type": "number"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scores", "avg"],
        }
        try:
            raw = _call_ollama_generate(qprompt, json_format=qschema, timeout=60.0, model=ctx.ollama_model)
            if raw:
                data = json.loads(raw)
                avg = float(data.get("avg", 3.0))
                quality_scores.append(avg)
                scores = data.get("scores", [])
                for i, sc in enumerate(scores):
                    if sc <= 2 and i < len(sample):
                        low_quality_ids.append(sample[i]["idx"])
                _send(ctx, {"type": "translation_quality_check",
                            "batch_avg": round(avg, 2), "issues": data.get("issues", [])})
        except Exception as exc:
            logger.debug("Translation quality check failed (non-fatal): %s", exc)

    _progress(ctx, "translating", 82, f"Tłumaczenie sub-agenta ({ctx.detected_language}→pl)…")
    ctx.segments = translate_segments_to_polish(
        ctx.segments, ctx.detected_language, on_batch_done=on_batch_done, model=ctx.ollama_model
    )

    # ── Second pass: re-translate poor segments with improved prompt ──────────
    retranslated = 0
    if low_quality_ids:
        _progress(ctx, "translating", 94, f"Ponowne tłumaczenie {len(low_quality_ids)} słabych segmentów…")
        for idx in set(low_quality_ids):
            if idx >= len(ctx.segments):
                continue
            seg = ctx.segments[idx]
            orig = seg.get("text", "")
            # Build richer context: up to 2 preceding and 1 following translated segments
            ctx_before = [
                ctx.segments[j].get("text_pl") or ctx.segments[j].get("text", "")
                for j in range(max(0, idx - 2), idx)
                if ctx.segments[j].get("text_pl") or ctx.segments[j].get("text")
            ]
            ctx_after = []
            if idx + 1 < len(ctx.segments):
                nxt = ctx.segments[idx + 1].get("text_pl") or ctx.segments[idx + 1].get("text", "")
                if nxt:
                    ctx_after = [nxt]

            retry_prompt = (
                f"Przetłumacz poniższe zdanie z języka {ctx.detected_language} na naturalny, "
                "płynny POLSKI. Zachowaj styl i ton. Nie dodawaj nic od siebie.\n"
                "Odpowiedz TYLKO polskim tłumaczeniem.\n\n"
                + (f"Poprzedni kontekst: {' / '.join(ctx_before)}\n" if ctx_before else "")
                + f"TŁUMACZ TO: {orig}\n"
                + (f"Następny kontekst: {ctx_after[0]}\n" if ctx_after else "")
            )
            try:
                new_pl = _call_ollama_generate(
                    retry_prompt, timeout=30.0, model=ctx.ollama_model
                )
                if new_pl and new_pl.strip() and new_pl.strip().lower() != orig.lower():
                    ctx.segments[idx]["text_pl"] = new_pl.strip()
                    retranslated += 1
                    _send(ctx, {"type": "translation_chunk",
                                "updates": [{"idx": idx, "text_pl": ctx.segments[idx]["text_pl"]}]})
            except Exception as exc:
                logger.debug("Retranslation of segment %d failed: %s", idx, exc)

    overall_avg = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None
    translated = sum(1 for s in ctx.segments if s.get("text_pl"))
    return {
        "segments_translated": translated,
        "source_language": ctx.detected_language,
        "quality_checks_performed": len(quality_scores),
        "overall_avg_quality": overall_avg,
        "low_quality_retranslated": retranslated,
        "quality_acceptable": (overall_avg is None or overall_avg >= 3.0),
    }


# ── New tools ─────────────────────────────────────────────────────────────────

def _tool_analyze_audio_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Analyse audio signal: RMS, dynamic range, clipping, silence ratio, noise floor."""
    if ctx.audio_pcm is None:
        return {"error": "Audio not decoded. Run get_audio_info first."}

    audio = ctx.audio_pcm
    sr = ctx.sample_rate

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    dynamic_range_db = round(20 * np.log10(peak / (rms + 1e-10)), 1)
    clipping_ratio = float(np.mean(np.abs(audio) > 0.98))

    # Frame-based energy for silence / noise estimation
    frame_size = int(0.025 * sr)  # 25 ms
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, frame_size)]
    if frames:
        energies = np.array([np.mean(f ** 2) for f in frames])
        noise_floor_db = round(10 * np.log10(float(np.percentile(energies, 10)) + 1e-10), 1)
        silence_threshold = float(np.percentile(energies, 20))
        silence_ratio = round(float(np.mean(energies < silence_threshold)), 3)
    else:
        noise_floor_db = -60.0
        silence_ratio = 0.0

    warnings: List[str] = []
    recommendations: List[str] = []
    if clipping_ratio > 0.001:
        warnings.append(f"Audio clipping: {clipping_ratio*100:.2f}% of samples saturated")
    if rms < 0.01:
        warnings.append("Very low volume level")
        recommendations.append("Use vad_filter_threshold=0.10 or 0.15 — lower sensitivity for quiet audio")
    if silence_ratio > 0.65:
        warnings.append(f"High silence ratio ({silence_ratio:.0%}) — long pauses or sparse speech")
        recommendations.append("Increase min_silence_duration_ms=600 and min_speech_duration_ms=200")
    if dynamic_range_db < 6:
        warnings.append("Low dynamic range — compressed or phone-call audio")
        recommendations.append("Consider beam_size=7 for compressed audio")

    return {
        "rms_level": round(rms, 5),
        "peak_level": round(peak, 4),
        "dynamic_range_db": dynamic_range_db,
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": silence_ratio,
        "noise_floor_db": noise_floor_db,
        "duration_sec": round(ctx.duration, 1),
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _tool_validate_translation_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    LLM-based sample quality assessment of the current translations.
    Scores 1-5 per segment sample; returns worst segment IDs for targeted retry.
    """
    translated = [s for s in ctx.segments if s.get("text_pl") and s.get("text_pl") != s.get("text")]
    if not translated:
        return {"skipped": True, "reason": "No translations yet. Run translate_to_polish first."}

    import random
    n_sample = min(int(args.get("sample_size", 6)), 10, len(translated))
    sample = random.sample(translated, n_sample)

    src_lang = ctx.detected_language
    lines = "\n".join(
        f"{i+1}. [{src_lang}] {s.get('text','')[:150]}\n   [pl]   {s.get('text_pl','')[:150]}"
        for i, s in enumerate(sample)
    )
    prompt = (
        f"Jesteś ekspertem od tłumaczeń. Oceń poniższe tłumaczenia z {src_lang} na polski.\n"
        f"Skala: 1=kompletnie błędne/inny język, 2=bardzo złe, 3=akceptowalne, 4=dobre, 5=świetne.\n\n"
        f"{lines}\n\n"
        "Zwróć JSON: {\"scores\": [int,...], \"avg_score\": float, "
        "\"worst_indices\": [int,...], \"issues\": [str,...]}"
    )
    json_schema = {
        "type": "object",
        "properties": {
            "scores": {"type": "array", "items": {"type": "integer"}},
            "avg_score": {"type": "number"},
            "worst_indices": {"type": "array", "items": {"type": "integer"}},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scores", "avg_score"],
    }
    raw = _call_ollama_generate(prompt, json_format=json_schema, timeout=90.0, model=ctx.ollama_model)
    if not raw:
        return {"error": "LLM did not respond"}

    try:
        data = json.loads(raw)
        avg = float(data.get("avg_score", 0.0))
        worst_local = [i for i in data.get("worst_indices", []) if i < len(sample)]
        worst_ids = [sample[i].get("id", sample[i].get("idx", 0)) for i in worst_local]
        return {
            "sample_size": n_sample,
            "avg_score": round(avg, 2),
            "scores": data.get("scores", []),
            "issues": data.get("issues", []),
            "worst_segment_ids": worst_ids,
            "needs_retranslation": avg < 3.0,
            "recommendation": (
                f"Call retranslate_segments with segment_ids={worst_ids}"
                if avg < 3.5 and worst_ids else "Translation quality acceptable"
            ),
        }
    except (json.JSONDecodeError, KeyError) as exc:
        return {"error": f"Parse error: {exc}", "raw": raw[:200]}


def _tool_retranslate_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Re-translate specific segments (by id) with an improved prompt.
    Use after validate_translation_quality identifies weak segments.
    """
    segment_ids: List[int] = [int(x) for x in (args.get("segment_ids") or [])]
    temperature = float(args.get("temperature", 0.3))  # slightly creative for quality
    if not segment_ids:
        return {"error": "segment_ids required"}
    if not ctx.segments:
        return {"error": "No segments"}

    id_map = {s.get("id", i): i for i, s in enumerate(ctx.segments)}
    retranslated: List[int] = []
    src_lang = ctx.detected_language

    for seg_id in segment_ids:
        seg_idx = id_map.get(seg_id)
        if seg_idx is None:
            continue
        seg = ctx.segments[seg_idx]
        orig = seg.get("text", "")
        context_prev = ctx.segments[seg_idx - 1].get("text", "") if seg_idx > 0 else ""
        context_next = ctx.segments[seg_idx + 1].get("text", "") if seg_idx < len(ctx.segments) - 1 else ""

        prompt = (
            f"Przetłumacz poniższe zdanie z {src_lang} na NATURALNY POLSKI.\n"
            "Weź pod uwagę kontekst (zdania przed i po).\n"
            "Odpowiedz TYLKO gotowym polskim tłumaczeniem bez żadnych komentarzy.\n\n"
            + (f"Poprzednie zdanie (kontekst): {context_prev}\n" if context_prev else "")
            + f"ZDANIE DO PRZETŁUMACZENIA: {orig}\n"
            + (f"Następne zdanie (kontekst): {context_next}\n" if context_next else "")
        )
        try:
            body: Dict[str, Any] = {
                "model": ctx.ollama_model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": 4096, "temperature": temperature},
            }
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(f"{OLLAMA_URL}/api/generate", json=body)
                resp.raise_for_status()
                new_pl = resp.json().get("response", "").strip()
            if new_pl:
                ctx.segments[seg_idx]["text_pl"] = new_pl
                retranslated.append(seg_id)
                _send(ctx, {"type": "translation_chunk", "updates": [{"idx": seg_idx, "text_pl": new_pl}]})
        except Exception as exc:
            logger.warning("Retranslation segment %d failed: %s", seg_id, exc)

    return {"retranslated_count": len(retranslated), "segment_ids": retranslated}


def _tool_detect_topics(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Time-windowed topic detection via LLM. Identifies main topics per N-minute window.
    Useful for generating chapter markers or navigation.
    """
    if not ctx.segments:
        return {"error": "No segments. Run transcribe_audio first."}

    window_min = float(args.get("window_minutes", 5.0))
    window_sec = window_min * 60
    max_windows = int(args.get("max_windows", 8))

    # Build time windows
    windows: List[Dict] = []
    current_segs: List[str] = []
    current_start = 0.0
    prev_end = 0.0

    for seg in ctx.segments:
        t = seg.get("start") or 0
        if t - current_start >= window_sec and current_segs:
            windows.append({"start": current_start, "end": prev_end,
                            "text": " ".join(current_segs)[:1000]})
            current_segs = []
            current_start = t
            if len(windows) >= max_windows:
                break
        current_segs.append((seg.get("text_pl") or seg.get("text") or "").strip())
        prev_end = seg.get("end") or t

    if current_segs:
        windows.append({"start": current_start, "end": prev_end,
                        "text": " ".join(current_segs)[:1000]})

    _progress(ctx, "extracting", 75, f"Wykrywanie tematów ({len(windows)} okien)…")

    topics: List[Dict] = []
    for win in windows[:max_windows]:
        prompt = (
            "Podaj główny temat poniższego fragmentu rozmowy w 1 krótkim zdaniu (max 15 słów) po polsku.\n\n"
            f"Fragment:\n{win['text']}\n\nTemat:"
        )
        try:
            topic = _call_ollama_generate(prompt, timeout=30.0, model=ctx.ollama_model)
            topics.append({
                "start_sec": round(win["start"], 1),
                "end_sec": round(win["end"], 1),
                "topic": topic.strip()[:120] if topic else "—",
            })
        except Exception:
            topics.append({"start_sec": round(win["start"], 1), "end_sec": round(win["end"], 1), "topic": "—"})

    ctx.topics = topics  # store so the final result event carries topics to the UI
    return {"windows": len(topics), "topics": topics}


def _tool_retranscribe_time_range(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Re-transcribe a specific time range and replace/supplement existing segments.
    Useful when verify_transcript_quality reveals a problematic section.
    """
    if ctx.audio_pcm is None:
        return {"error": "Audio not decoded. Run get_audio_info first."}

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available for retranscription"}
    models.get("ensure_whisper_gpu", lambda: None)()

    start_sec = float(args.get("start_sec", 0.0))
    end_sec = float(args.get("end_sec", min(start_sec + 60, ctx.duration)))
    language = args.get("language") or ctx.language_hint
    extra_kw = args.get("params") or {}

    sr = ctx.sample_rate
    start_smp = int(start_sec * sr)
    end_smp = int(end_sec * sr)
    fragment = ctx.audio_pcm[start_smp:end_smp]

    _progress(ctx, "transcribing", 5, f"Re-transkrypcja {start_sec:.0f}–{end_sec:.0f}s…")

    try:
        new_chunks, detected_lang, _ = whisper.transcribe(
            fragment, language, progress_cb=lambda *a: None,
            extra_kw=extra_kw if extra_kw else None,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if not new_chunks:
        return {"replaced": 0, "note": "No segments returned from re-transcription"}

    # Adjust timestamps to absolute
    for ch in new_chunks:
        ch["start"] = round(ch["start"] + start_sec, 3)
        ch["end"] = round(ch["end"] + start_sec, 3)
        for w in (ch.get("words") or []):
            w["start"] = round(w["start"] + start_sec, 3)
            w["end"] = round(w["end"] + start_sec, 3)

    # Remove existing segments in this time range
    original_count = len(ctx.segments)
    ctx.segments = [s for s in ctx.segments
                    if not (start_sec <= (s.get("start") or 0) < end_sec)]

    # Insert new segments and sort
    ctx.segments.extend(new_chunks)
    ctx.segments.sort(key=lambda s: s.get("start") or 0)

    # Re-index
    for i, s in enumerate(ctx.segments):
        s["id"] = i

    return {
        "replaced_range": f"{start_sec:.1f}–{end_sec:.1f}s",
        "segments_removed": original_count - (len(ctx.segments) - len(new_chunks)),
        "new_segments": len(new_chunks),
        "total_segments": len(ctx.segments),
        "detected_language": detected_lang,
    }


def _tool_identify_speakers(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    from ..speaker_identifier import identify_speakers
    _progress(ctx, "identifying", 94, "Identyfikacja mówców (LLM)…")
    # Pass windowed transcript so speaker_identifier doesn't overload context
    try:
        ctx.display_names = identify_speakers(
            ctx.segments[-ctx.max_ctx_segments:],
            ctx.speaker_profiles_raw,
        )
    except Exception as exc:
        logger.warning("Speaker identification failed: %s", exc)
        ctx.display_names = {}

    # ── Emit speaker_profiles_update so UI immediately applies display names ──
    # Without this event the transcript keeps showing GŁOS_01 etc. until the
    # final 'result' event arrives at the very end.
    if ctx.display_names:
        speaker_profiles = _build_speaker_profiles(ctx)
        _send(ctx, {
            "type": "speaker_profiles_update",
            "speaker_profiles": speaker_profiles,
            "display_names": ctx.display_names,
        })

    return {"display_names": ctx.display_names}


def _tool_emit_partial_result(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Stream current segments to UI without finishing processing."""
    if not ctx.segments:
        return {"skipped": True, "reason": "No segments yet"}

    speaker_profiles = _build_speaker_profiles(ctx)
    segments_out = _build_segments_out(ctx)

    _send(ctx, {
        "type": "partial_segments",
        "segments": segments_out,
        "speaker_profiles": speaker_profiles,
        "detected_language": ctx.detected_language,
        "duration": round(ctx.duration, 2),
        "message": args.get("message", "Podgląd wstępny…"),
    })
    ctx.partial_emitted = True
    return {"emitted": len(segments_out), "message": args.get("message", "")}


def _tool_extract_entities(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")

    # Use windowed context to avoid LLM overload on long transcripts
    full_text = " ".join((s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments)
    WINDOW_CHARS = ctx.llm_chunk_chars
    OVERLAP = min(600, WINDOW_CHARS // 10)
    MAX_WINDOWS = 6
    windows: List[str] = []
    if len(full_text) <= WINDOW_CHARS:
        windows = [full_text]
    else:
        step = WINDOW_CHARS - OVERLAP
        for i in range(0, len(full_text), step):
            if len(windows) >= MAX_WINDOWS:
                break
            windows.append(full_text[i: i + WINDOW_CHARS])

    lang_names = {"en": "angielski", "ru": "rosyjski", "uk": "ukraiński", "de": "niemiecki", "fr": "francuski"}
    lang_note = (f"Tekst pochodzi z języka {lang_names.get(ctx.detected_language, ctx.detected_language)}. "
                 if ctx.detected_language not in ("pl", "auto") else "")

    _progress(ctx, "extracting", 80, f"Ekstrakcja encji ({len(windows)} okien)…")
    merged: Dict[str, set] = {k: set() for k in ("persons", "organizations", "locations", "dates", "keywords")}
    json_schema = {
        "type": "object",
        "properties": {k: {"type": "array", "items": {"type": "string"}} for k in merged},
        "required": list(merged.keys()),
    }

    for wi, window in enumerate(windows):
        if len(windows) > 1:
            _progress(ctx, "extracting", 80 + (wi * 4 // len(windows)), f"Encje okno {wi+1}/{len(windows)}…")
        prompt = (f"{lang_note}Przeanalizuj tekst i wyodrębnij encje. Odpowiedz WYŁĄCZNIE w JSON.\n\n"
                  f"Tekst:\n{window}\n\n"
                  "Zwróć JSON: persons, organizations, locations, dates, keywords (tablice stringów).")
        raw = _call_ollama_generate(
            prompt, json_format=json_schema, timeout=120.0,
            model=ctx.ollama_model, num_predict=400,  # entity lists are short
        )
        if raw:
            try:
                data = json.loads(raw)
                for k in merged:
                    for v in data.get(k, []):
                        if v:
                            merged[k].add(v.strip())
            except (json.JSONDecodeError, AttributeError):
                pass

    ctx.entities = {k: sorted(v) for k, v in merged.items()}
    ctx.entities["keywords"] = ctx.entities["keywords"][:20]
    return {k: len(v) for k, v in ctx.entities.items()}


def _tool_build_rag_index(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    _progress(ctx, "embedding", 88, "Budowanie indeksu RAG…")

    CHUNK_CHARS, OVERLAP_CHARS, MAX_CHUNKS = 800, 150, 300
    texts: List[str] = []
    chunk_meta: List[Dict] = []
    buf = ""
    buf_seg_ids: List[int] = []
    buf_start = buf_end = buf_speaker = None

    full_seg_texts = [(s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments]
    for i, seg in enumerate(ctx.segments):
        txt = full_seg_texts[i]
        if not txt:
            continue
        if not buf:
            buf_start = seg.get("start")
            buf_speaker = seg.get("speaker")
        buf += (" " if buf else "") + txt
        buf_seg_ids.append(seg.get("id", i))
        buf_end = seg.get("end")
        if len(buf) >= CHUNK_CHARS:
            texts.append(buf.strip())
            chunk_meta.append({"segmentIds": buf_seg_ids[:], "start": buf_start, "end": buf_end, "speaker": buf_speaker})
            if len(texts) >= MAX_CHUNKS:
                break
            buf = buf[-OVERLAP_CHARS:]
            buf_seg_ids = buf_seg_ids[-3:]
            buf_start = buf_end

    if buf.strip() and len(texts) < MAX_CHUNKS:
        texts.append(buf.strip())
        chunk_meta.append({"segmentIds": buf_seg_ids[:], "start": buf_start, "end": buf_end, "speaker": buf_speaker})

    if not texts:
        return {"chunks": 0, "skipped": True}

    all_embeddings: List[List[float]] = []
    for b in range(0, len(texts), 32):
        vecs = _call_ollama_embed(texts[b: b + 32], timeout=120.0, embed_model=ctx.ollama_model)
        all_embeddings.extend(vecs if vecs else [[0.0] * 768] * len(texts[b: b + 32]))

    ctx.rag_entries = [
        {"id": i, "text": texts[i], "embedding": all_embeddings[i] if i < len(all_embeddings) else [], "metadata": chunk_meta[i]}
        for i in range(len(texts))
    ]
    return {"chunks": len(ctx.rag_entries), "embedding_model": OLLAMA_EMBEDDING_MODEL}


def _tool_summarize_transcript(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")

    style = args.get("style", "structured")
    full_text = " ".join((s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments)
    duration_min = round(ctx.duration / 60)
    is_long = ctx.duration > 30 * 60
    WINDOW_CHARS, OVERLAP_CHARS, MAX_WINDOWS = 6000, 600, 6

    lang_names = {"en": "angielski", "ru": "rosyjski", "uk": "ukraiński", "de": "niemiecki"}
    lang_note = (f" (język oryginalny: {lang_names.get(ctx.detected_language, ctx.detected_language)})"
                 if ctx.detected_language not in ("pl", "auto") else "")

    # Build speaker context block (gender, age, speech share, display name)
    speaker_context_lines: List[str] = []
    if ctx.speaker_profiles_raw or ctx.display_names:
        # Compute per-speaker speech time share
        sp_time: Dict[str, float] = {}
        for seg in ctx.segments:
            sp = seg.get("speaker", "?")
            sp_time[sp] = sp_time.get(sp, 0.0) + (seg.get("end") or 0) - (seg.get("start") or 0)
        total_sp_time = max(sum(sp_time.values()), 1.0)

        for sp, profile in ctx.speaker_profiles_raw.items():
            display = ctx.display_names.get(sp, sp)
            gender_map = {"meski": "mężczyzna", "zenski": "kobieta", "dziecko": "dziecko"}
            gender = gender_map.get(profile.get("gender") or "", "")
            age = profile.get("age_estimate")
            age_str = f" ~{int(age)} lat" if age else ""
            share = sp_time.get(sp, 0.0) / total_sp_time
            speaker_context_lines.append(
                f"- {display} ({sp}): {gender}{age_str}, {share:.0%} czasu mówienia"
            )

    speaker_context = ""
    if speaker_context_lines:
        speaker_context = "\n\nMówcy w nagraniu:\n" + "\n".join(speaker_context_lines)

    style_note = {
        "brief": "Napisz krótkie streszczenie (3-5 zdań).",
        "detailed": "Napisz szczegółowy raport.",
        "structured": (
            "Napisz raport:\n## Streszczenie\n(2-4 zdania)\n\n"
            "## Główne tematy\n(lista)\n\n"
            "## Kluczowe decyzje i ustalenia\n(lista lub Brak)\n\n"
            "## Uczestnicy rozmowy\n(imię/rola + wiek/płeć jeśli znane + język)\n\n"
            "## Następne kroki\n(lista lub Nie wspomniano)"
        ),
    }.get(style, "Napisz szczegółowy raport.")

    def _summary_prompt(text: str) -> str:
        return (
            f"Stwórz raport z transkrypcji rozmowy{lang_note}. PO POLSKU.{speaker_context}\n\n"
            f"Transkrypcja:\n{text}\n\n{style_note}\n\nRaport:"
        )

    _progress(ctx, "summarizing", 93, f"Podsumowanie{' (długie nagranie)' if is_long else ''}…")

    if not is_long or len(full_text) <= WINDOW_CHARS:
        report = _call_ollama_generate(
            _summary_prompt(full_text[:WINDOW_CHARS]), timeout=300.0, model=ctx.ollama_model
        )
    else:
        step = WINDOW_CHARS - OVERLAP_CHARS
        windows = [full_text[i: i + WINDOW_CHARS] for i in range(0, len(full_text), step)][:MAX_WINDOWS]
        partials: List[str] = []
        for wi, w in enumerate(windows):
            _progress(ctx, "summarizing", 93 + (wi * 3 // len(windows)),
                      f"Streszczenie fragment {wi+1}/{len(windows)}…")
            p = _call_ollama_generate(
                _summary_prompt(w), timeout=300.0, model=ctx.ollama_model
            )
            if p:
                partials.append(f"### Fragment {wi+1}\n{p}")
                # Emit incremental partial summary so UI shows progress
                partial_preview = f"**Fragment {wi+1}/{len(windows)} opracowany.**\n\n{p[:800]}"
                _send(ctx, {
                    "type": "partial_summary",
                    "window_index": wi,
                    "window_count": len(windows),
                    "content": partial_preview,
                })

        if partials:
            _progress(ctx, "summarizing", 98, "Łączenie streszczeń…")
            combined = "\n\n".join(partials)[: WINDOW_CHARS * 2]
            reduce_prompt = (
                f"Częściowe streszczenia nagrania (~{duration_min} min)."
                f"{speaker_context}\n\n"
                "Stwórz jeden spójny raport PO POLSKU:\n\n"
                f"{combined}\n\n"
                "## Streszczenie\n## Główne tematy\n## Uczestnicy\nRaport:"
            )
            report = _call_ollama_generate(
                reduce_prompt, timeout=360.0, model=ctx.ollama_model
            )
        else:
            report = ""

    if report:
        ctx.report = report
        ctx.summary = " ".join(report.split("\n")[:3])[:300]

    return {"report_length": len(report) if report else 0}


def _tool_save_memory(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    observation = args.get("observation", "").strip()
    improvement = args.get("improvement", "").strip()
    if not observation or not improvement:
        return {"error": "observation and improvement are required"}
    mem = _save_memory_fn(observation, improvement, args.get("tags", []))
    _agent_event(ctx, "agent_memory", memory=mem)
    return {"saved": True, "id": mem.get("id")}


def _tool_finish(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    return {
        "complete": True,
        "message": args.get("message", "Przetwarzanie zakończone."),
        "segment_count": len(ctx.segments),
        "detected_language": ctx.detected_language,
        "has_translation": any(s.get("text_pl") for s in ctx.segments),
        "has_entities": ctx.entities is not None,
        "has_summary": ctx.summary is not None,
        "rag_chunks": len(ctx.rag_entries),
    }


# ── New quality & analysis tools ──────────────────────────────────────────────

def _tool_refine_speaker_assignments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Post-diarization speaker refinement using three strategies:
    1. Neighbour voting for short segments (<1.5s) — majority of ±4 neighbours wins.
    2. Micro-gap bridging — same speaker segments < gap_sec apart are merged.
    3. LLM-assisted ambiguity resolution — for segments where neighbours disagree,
       ask the LLM to infer speaker from textual cues (name mentions, pronouns).
    """
    if not ctx.segments:
        return {"error": "No segments. Run diarize_audio first."}

    from collections import Counter as _Counter

    gap_sec = float(args.get("gap_sec", 0.5))
    short_thresh = float(args.get("short_sec", 1.5))
    window = int(args.get("window", 4))

    changes = 0

    # Strategy 1: Neighbour majority vote for short segments
    for i, seg in enumerate(ctx.segments):
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur >= short_thresh:
            continue
        start_j = max(0, i - window)
        end_j = min(len(ctx.segments), i + window + 1)
        neighbours = [ctx.segments[j]["speaker"]
                      for j in range(start_j, end_j) if j != i]
        if not neighbours:
            continue
        top_sp, top_cnt = _Counter(neighbours).most_common(1)[0]
        if top_cnt >= max(2, len(neighbours) // 2) and top_sp != seg.get("speaker"):
            seg["speaker"] = top_sp
            changes += 1

    # Strategy 2: Bridge micro-gaps between same-speaker runs
    bridge_count = 0
    for i in range(1, len(ctx.segments) - 1):
        prev_sp = ctx.segments[i - 1].get("speaker")
        next_sp = ctx.segments[i + 1].get("speaker") if i + 1 < len(ctx.segments) else None
        curr_sp = ctx.segments[i].get("speaker")
        gap_before = (ctx.segments[i].get("start") or 0) - (ctx.segments[i - 1].get("end") or 0)
        gap_after = (ctx.segments[i + 1].get("start") if i + 1 < len(ctx.segments) else 999) - (ctx.segments[i].get("end") or 0)
        # If flanked by same speaker with tiny gaps, absorb
        if prev_sp == next_sp and prev_sp != curr_sp and gap_before < gap_sec and gap_after < gap_sec:
            ctx.segments[i]["speaker"] = prev_sp
            bridge_count += 1
            changes += 1

    # Strategy 3: LLM disambiguation for remaining ambiguous short segments
    ambiguous = []
    for i, seg in enumerate(ctx.segments):
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur < 0.8:
            text = (seg.get("text_pl") or seg.get("text") or "").strip()
            if text:
                # Build context window
                context_segs = ctx.segments[max(0, i-2):i] + ctx.segments[i+1:min(len(ctx.segments), i+3)]
                context_text = "\n".join(
                    f"[{s['speaker']}] {(s.get('text_pl') or s.get('text') or '').strip()}"
                    for s in context_segs
                )
                ambiguous.append((i, seg["speaker"], text, context_text))

    if ambiguous and len(ambiguous) <= 8:  # keep prompt small — gemma4:26b is slow on long inputs
        prompt_lines = []
        for i, (idx, current_sp, text, context) in enumerate(ambiguous):
            speakers = list({s.get("speaker") for s in ctx.segments if s.get("speaker")})
            prompt_lines.append(
                f"{i+1}. Krótki segment [{current_sp}]: \"{text}\"\n"
                f"   Kontekst:\n{context}\n"
                f"   Dostępni mówcy: {speakers}"
            )
        prompt = (
            "Który mówca powiedział poniższe krótkie segmenty? "
            "Skorzystaj z kontekstu (nazwy, tytuły, zaimki, zwroty).\n\n"
            + "\n\n".join(prompt_lines)
            + "\n\nZwróć JSON: {\"assignments\": [{\"index\": int, \"speaker\": str}, ...]}"
        )
        schema = {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "speaker": {"type": "string"},
                        },
                        "required": ["index", "speaker"],
                    },
                }
            },
            "required": ["assignments"],
        }
        try:
            raw = _call_ollama_generate(prompt, json_format=schema, timeout=60.0, model=ctx.ollama_model)
            if raw:
                data = json.loads(raw)
                valid_speakers = {s.get("speaker") for s in ctx.segments}
                for item in data.get("assignments", []):
                    local_idx = int(item.get("index", -1)) - 1
                    new_sp = item.get("speaker", "")
                    if 0 <= local_idx < len(ambiguous) and new_sp in valid_speakers:
                        seg_idx = ambiguous[local_idx][0]
                        if ctx.segments[seg_idx]["speaker"] != new_sp:
                            ctx.segments[seg_idx]["speaker"] = new_sp
                            changes += 1
        except Exception as exc:
            logger.debug("LLM speaker disambiguation failed (non-fatal): %s", exc)

    return {
        "changes": changes,
        "short_segments_checked": len([s for s in ctx.segments if (s.get("end", 0) - s.get("start", 0)) < short_thresh]),
        "bridges": bridge_count,
        "llm_disambiguated": len(ambiguous),
    }


def _tool_extract_quotes_and_facts(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Extract structured quotes, facts, decisions, and key questions from the transcript.
    Results stored in ctx and included in the final result event.
    Critical for analysis quality — quotes provide verbatim evidence.
    """
    if not ctx.segments:
        return {"error": "No segments. Run transcribe_audio first."}

    # Skip for short recordings — not worth the LLM time
    if ctx.duration < 5 * 60:
        return {"skipped": True, "reason": f"Recording too short ({ctx.duration:.0f}s < 5 min) — skipping quote extraction"}

    def fmt_time(s: float) -> str:
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m}:{sec:02d}"

    # Build labelled transcript — cap at 60 segments / 3000 chars for speed
    lines: List[str] = []
    for seg in ctx.segments[:60]:
        text = (seg.get("text_pl") or seg.get("text") or "").strip()
        if not text:
            continue
        sp = seg.get("speaker", "?")
        display = ctx.display_names.get(sp, sp)
        t = fmt_time(float(seg.get("start") or 0))
        lines.append(f"[{t} {display}] {text}")

    transcript = "\n".join(lines)[:3000]

    prompt = (
        "Przeanalizuj transkrypt i wyodrębnij:\n"
        "1. CYTATY — max 4 znaczące dosłowne wypowiedzi.\n"
        "2. FAKTY — max 5 konkretnych twierdzeń, liczb, dat.\n"
        "3. DECYZJE — co ustalono (lub puste).\n"
        "4. KLUCZOWE PYTANIA — max 3 ważne pytania.\n\n"
        f"Transkrypt:\n{transcript}\n\n"
        "JSON:"
    )
    schema = {
        "type": "object",
        "properties": {
            "quotes": {"type": "array", "items": {"type": "object"}},
            "facts": {"type": "array", "items": {"type": "object"}},
            "decisions": {"type": "array", "items": {"type": "object"}},
            "key_questions": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["quotes", "facts"],
    }

    _progress(ctx, "extracting", 85, "Ekstrakcja cytatów i faktów…")
    # num_predict=800 caps output — prevents 5+ minute JSON generation runs
    raw = _call_ollama_generate(
        prompt, json_format=schema, timeout=120.0,
        model=ctx.ollama_model, num_predict=800,
    )

    if not raw:
        return {"error": "LLM did not respond"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse quotes/facts response"}

    # Store in context
    ctx._quotes_and_facts = data  # type: ignore[attr-defined]

    # Inject quotes as high-priority RAG entries (prepend to rag_entries)
    for i, q in enumerate(data.get("quotes", [])[:15]):
        text = q.get("text", "")
        speaker = q.get("speaker", "")
        ts = q.get("timestamp", "")
        if text:
            # Find matching segment for timestamps
            ctx.rag_entries.insert(i, {
                "id": -(i + 1),  # negative IDs mark special entries
                "text": f"CYTAT [{ts} {speaker}]: {text}",
                "embedding": [],  # will be recomputed if build_rag_index is called again
                "metadata": {"type": "quote", "speaker": speaker, "timestamp": ts},
            })

    return {
        "quotes_extracted": len(data.get("quotes", [])),
        "facts_extracted": len(data.get("facts", [])),
        "decisions_extracted": len(data.get("decisions", [])),
        "key_questions_extracted": len(data.get("key_questions", [])),
        "quotes_sample": [q.get("text", "")[:80] for q in data.get("quotes", [])[:3]],
    }


def _tool_diarize_first_transcribe(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Diarize-first pipeline with multi-pass validation and hallucination filtering.

    Pipeline:
    1. Run pyannote on full audio → speaker turns (including <1s interjections).
    2. Sort turns, detect overlapping regions, annotate with `overlapping` flag.
    3. Process in batches of batch_size turns:
       a. Transcribe each turn with Whisper (with context from previous turns).
       b. For short/uncertain turns: also transcribe WITHOUT context and compare.
          If results agree → high confidence. If differ → use better one / discard.
       c. Post-turn validation: confidence score, words-per-second rate, no-speech prob.
       d. Hallucination filter: remove turns whose text matches known patterns
          (only when combined with very low confidence).
    4. Emit streaming updates after each batch.
    5. De-duplicate: if same or very similar text appears 3+ consecutive times
       in the same speaker → keep only the first occurrence.

    Result: fine-grained segments with speaker attribution, overlap annotation,
    confidence scores, and hallucinations removed.
    """
    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    diarizer = models.get("diarizer")
    whisper = models.get("whisper")
    if diarizer is None or diarizer._method == "none":
        return {"error": "Diarizer not available"}
    if whisper is None:
        return {"error": "Whisper not available"}

    models.get("ensure_whisper_gpu", lambda: None)()

    num_speakers    = int(args.get("num_speakers") or 0)
    context_pad     = float(args.get("context_padding_sec") or 0.5)
    language        = args.get("language") or ctx.language_hint or ctx.detected_language
    min_dur         = float(args.get("min_turn_sec") or 0.1)
    conf_threshold  = float(args.get("min_confidence") or 0.30)  # drop below this
    batch_size      = int(args.get("batch_size") or 20)
    multi_pass      = bool(args.get("multi_pass") if args.get("multi_pass") is not None else True)

    sr = ctx.sample_rate
    total_dur = ctx.duration

    # ── Step 1: Multi-pass pyannote diarization ───────────────────────────────
    # Run pyannote with num_speakers=N AND num_speakers=N+1 when N is provided.
    # Pick the run where the smallest speaker has ≥5% of speaking time — this
    # prevents under-counting speakers who appear briefly.
    _progress(ctx, "diarizing", 5, "Diaryzacja pełnego nagrania (pyannote, multi-pass)…")

    import tempfile, os as _os
    import soundfile as _sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _sf.write(f.name, ctx.audio_pcm, sr)
        tmp_path = f.name

    from pyannote.core import Annotation

    def _run_pyannote(n_spk: Optional[int]) -> List[Dict]:
        """Run pyannote and return sorted raw turns."""
        kw: Dict = {}
        if n_spk and n_spk > 0:
            kw["num_speakers"] = n_spk
        res = diarizer._pipeline(tmp_path, **kw)
        if hasattr(res, "speaker_diarization"):
            ann = res.speaker_diarization
        elif isinstance(res, Annotation):
            ann = res
        else:
            ann = res
        return sorted([
            {"start": float(t.start), "end": float(t.end), "speaker_raw": sp}
            for t, _, sp in ann.itertracks(yield_label=True)
            if float(t.end) - float(t.start) >= min_dur
        ], key=lambda x: x["start"])

    def _min_speaker_share(turns: List[Dict]) -> float:
        """Fraction of speaking time held by the smallest speaker."""
        from collections import defaultdict as _dd2
        sp_t: Dict[str, float] = _dd2(float)
        total = 0.0
        for t in turns:
            dur = t["end"] - t["start"]
            sp_t[t["speaker_raw"]] += dur
            total += dur
        if total < 1e-6 or not sp_t:
            return 0.0
        return min(sp_t.values()) / total

    try:
        turns_n = _run_pyannote(num_speakers if num_speakers > 0 else None)

        # Multi-pass: also try with one more speaker and keep the better result
        if num_speakers > 0 and len(set(t["speaker_raw"] for t in turns_n)) < num_speakers + 1:
            _progress(ctx, "diarizing", 10, f"Diaryzacja pass 2 (num_speakers={num_speakers+1})…")
            try:
                turns_np1 = _run_pyannote(num_speakers + 1)
                # If N+1 gives a speaker with ≥ 5% share, use that result
                if _min_speaker_share(turns_np1) >= 0.05:
                    turns_n = turns_np1
                    logger.info("Multi-pass diarization: using N+1=%d speakers (min share ≥5%%)", num_speakers + 1)
            except Exception as exc:
                logger.debug("Multi-pass diarization N+1 failed (non-fatal): %s", exc)

        raw_turns = turns_n
    finally:
        try: _os.unlink(tmp_path)
        except OSError: pass

    if not raw_turns:
        return {"error": "Pyannote returned no speaker turns"}

    # Map raw IDs → GŁOS_XX by speaking time
    from collections import defaultdict as _dd
    sp_time: Dict[str, float] = _dd(float)
    for t in raw_turns:
        sp_time[t["speaker_raw"]] += t["end"] - t["start"]
    sorted_sp = sorted(sp_time, key=sp_time.__getitem__, reverse=True)
    sp_map = {sp: f"GŁOS_{i+1:02d}" for i, sp in enumerate(sorted_sp)}

    # ── Step 2: Overlap detection ─────────────────────────────────────────────
    # Mark turns that overlap with the previous turn (different speaker = crosstalk)
    for i in range(1, len(raw_turns)):
        prev = raw_turns[i - 1]
        curr = raw_turns[i]
        if curr["start"] < prev["end"]:
            overlap_sec = prev["end"] - curr["start"]
            curr["overlapping"] = True
            curr["overlap_with"] = sp_map.get(prev["speaker_raw"], prev["speaker_raw"])
            curr["overlap_sec"] = round(overlap_sec, 3)
        else:
            curr["overlapping"] = False

    _progress(ctx, "diarizing", 15,
              f"Pyannote: {len(sorted_sp)} mówców, {len(raw_turns)} tur → transkrypcja w partiach…")

    # ── Step 3: Hallucination filter (general, signal-based — no phrase lists) ──
    # Whisper hallucinates on silence/noise. We detect this with two universal,
    # language-agnostic signals: (1) low average word confidence, and
    # (2) an impossible word rate for the turn duration. No curated phrase list
    # (those over-fit to specific recordings).

    def _avg_conf(word_list: List[Dict]) -> float:
        if not word_list:
            return 1.0
        return sum(w.get("probability", 1.0) for w in word_list) / len(word_list)

    def _words_per_sec(text: str, dur: float) -> float:
        return len(text.split()) / max(dur, 0.05)

    def _is_hallucination(text: str, words: List[Dict], dur: float) -> tuple:
        """Return (is_hallucination, reason). Signal-based only."""
        conf = _avg_conf(words)
        wps = _words_per_sec(text, dur)

        # Impossible word rate for the duration → hallucinated run
        if wps > 12 and dur < 1.0:
            return True, f"word_rate_too_high({wps:.0f}w/s)"

        # Essentially no acoustic evidence
        if conf < 0.12 and dur < 2.0:
            return True, f"very_low_conf({conf:.2f})"

        return False, ""

    def _jaccard_words(text_a: str, text_b: str) -> float:
        wa = set(text_a.lower().split())
        wb = set(text_b.lower().split())
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # ── Step 4: Transcription in batches ──────────────────────────────────────
    segments: List[Dict[str, Any]] = []
    hallucinations_removed = 0
    multi_pass_overrides = 0
    prev_text = ""
    n = len(raw_turns)

    for batch_start in range(0, n, batch_size):
        batch = raw_turns[batch_start: batch_start + batch_size]

        for turn in batch:
            t_start  = turn["start"]
            t_end    = turn["end"]
            dur      = t_end - t_start
            speaker  = sp_map[turn["speaker_raw"]]
            is_overlap = turn.get("overlapping", False)

            # Audio fragment with context padding
            pad_start = max(0.0, t_start - context_pad)
            pad_end   = min(total_dur, t_end + context_pad)
            frag = ctx.audio_pcm[int(pad_start * sr): int(pad_end * sr)]

            if len(frag) < int(0.05 * sr):
                continue

            # Base Whisper params — tuned for this turn's duration
            is_short = dur < 0.8
            kw_base: Dict[str, Any] = {
                "beam_size": 3 if is_short else 5,
                "best_of":   1 if is_short else 5,
                "word_timestamps": True,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.25 if is_short else 0.55,
                "compression_ratio_threshold": 2.4,
                "vad_filter": dur > 0.25,
                "vad_parameters": {
                    "threshold": 0.12 if is_short else 0.22,
                    "min_silence_duration_ms": 80,
                    "min_speech_duration_ms": 40,
                    "speech_pad_ms": 60,
                },
            }
            if language and language != "auto":
                kw_base["language"] = language

            def _run_whisper(kw_extra: Dict = {}) -> tuple:
                kw = {**kw_base, **kw_extra}
                try:
                    res = whisper.model.transcribe(frag, **kw)
                    gen = res[0] if isinstance(res, tuple) else res
                    all_texts, all_words = [], []
                    for seg in gen:
                        t = seg.text.strip()
                        if not t:
                            continue
                        all_texts.append(t)
                        if seg.words:
                            for w in seg.words:
                                all_words.append({
                                    "text": w.word,
                                    "start": float(w.start) + pad_start,
                                    "end": float(w.end) + pad_start,
                                    "probability": float(w.probability),
                                })

                    # ── CRITICAL: filter to actual turn boundaries ──────────────
                    # Whisper transcribes the PADDED fragment and may include words
                    # that belong to adjacent speaker turns (the padding region).
                    # Only keep words whose timestamp falls within [t_start, t_end].
                    # Add a small tolerance (30ms) for words that straddle the boundary.
                    BOUNDARY_TOL = 0.03  # 30 ms
                    turn_words = [
                        w for w in all_words
                        if w["start"] >= t_start - BOUNDARY_TOL
                        and w["start"] < t_end + BOUNDARY_TOL
                    ]

                    if turn_words:
                        # Rebuild text from in-turn words only
                        turn_text = "".join(w["text"] for w in turn_words).strip()
                        return turn_text, turn_words

                    if all_words:
                        # No word timestamps fell inside the turn — this commonly
                        # happens when the speech is entirely in the padding region.
                        # Return text only (no words) so confidence checks can still
                        # run, but avoid attaching mis-timestamped words to the segment.
                        return " ".join(all_texts).strip(), []

                    return " ".join(all_texts).strip(), []
                except Exception as exc:
                    logger.debug("Whisper turn failed: %s", exc)
                    return "", []

            # ── Pass 1: with context prompt ──────────────────────────────
            ctx_kw = {"initial_prompt": prev_text[-120:]} if prev_text else {}
            text_with_ctx, words_with_ctx = _run_whisper(ctx_kw)
            chosen_text = text_with_ctx
            chosen_words = words_with_ctx

            # ── Pass 2 (multi-pass): without context, for uncertain short turns ──
            if multi_pass and is_short and text_with_ctx:
                text_no_ctx, words_no_ctx = _run_whisper({})
                if text_no_ctx and text_no_ctx != text_with_ctx:
                    # Pick the version with higher average word confidence
                    conf_with = _avg_conf(words_with_ctx)
                    conf_no   = _avg_conf(words_no_ctx)
                    agree = _jaccard_words(text_with_ctx, text_no_ctx)
                    if agree < 0.4:
                        # Results disagree — take the higher-confidence version
                        if conf_no > conf_with:
                            chosen_text = text_no_ctx
                            chosen_words = words_no_ctx
                            multi_pass_overrides += 1
                        # If they strongly disagree and both low-confidence → discard
                        if max(conf_with, conf_no) < conf_threshold and dur < 0.5:
                            hallucinations_removed += 1
                            continue

            # ── Pass 3 (wider window): if still low-confidence, retry with larger padding ──
            if multi_pass and chosen_words and _avg_conf(chosen_words) < 0.45 and dur < 3.0:
                # Try with 2× padding for broader context
                wide_start = max(0.0, t_start - context_pad * 2)
                wide_end   = min(total_dur, t_end + context_pad * 2)
                wide_frag  = ctx.audio_pcm[int(wide_start * sr): int(wide_end * sr)]
                if len(wide_frag) > len(frag):
                    try:
                        res_wide = whisper.model.transcribe(wide_frag, **{**kw_base, **ctx_kw})
                        gen_wide = res_wide[0] if isinstance(res_wide, tuple) else res_wide
                        all_words_wide = []
                        texts_wide = []
                        for seg in gen_wide:
                            t = seg.text.strip()
                            if t:
                                texts_wide.append(t)
                                if seg.words:
                                    for w in seg.words:
                                        all_words_wide.append({
                                            "text": w.word,
                                            "start": float(w.start) + wide_start,
                                            "end": float(w.end) + wide_start,
                                            "probability": float(w.probability),
                                        })
                        # Filter to turn boundaries (same as _run_whisper)
                        BOUNDARY_TOL = 0.03
                        words_wide = [w for w in all_words_wide
                                      if w["start"] >= t_start - BOUNDARY_TOL
                                      and w["start"] < t_end + BOUNDARY_TOL]
                        if not words_wide:
                            words_wide = all_words_wide  # fall back if no filtered words
                        text_wide = "".join(w["text"] for w in words_wide).strip() if words_wide else " ".join(texts_wide).strip()
                        if text_wide and _avg_conf(words_wide) > _avg_conf(chosen_words):
                            chosen_text = text_wide
                            chosen_words = words_wide
                            multi_pass_overrides += 1
                    except Exception:
                        pass

            if not chosen_text:
                continue

            # ── Hallucination filter ─────────────────────────────────────
            is_halluc, reason = _is_hallucination(chosen_text, chosen_words, dur)
            if is_halluc:
                hallucinations_removed += 1
                logger.debug("Hallucination removed at %.1fs: %s [%s]", t_start, chosen_text[:50], reason)
                continue

            # ── Low-confidence check ─────────────────────────────────────
            conf = _avg_conf(chosen_words)
            if conf < conf_threshold and dur < 1.5:
                hallucinations_removed += 1
                logger.debug("Low-conf removed at %.1fs: %s (conf=%.2f)", t_start, chosen_text[:50], conf)
                continue

            prev_text = chosen_text

            seg_dict: Dict[str, Any] = {
                "text": chosen_text,
                "start": t_start,
                "end": t_end,
                "speaker": speaker,
                "words": chosen_words,
            }
            if is_overlap:
                seg_dict["overlapping"] = True
                seg_dict["overlap_with"] = turn.get("overlap_with")
                seg_dict["overlap_sec"] = turn.get("overlap_sec")
            segments.append(seg_dict)

            # Streaming update for live UI
            _send(ctx, {
                "type": "segment_chunk",
                "segments": [{
                    "id": len(segments) - 1,
                    "start": t_start, "end": t_end,
                    "text": chosen_text, "speaker": speaker,
                    "language": language or "pl",
                }],
                "offset_sec": t_start,
                "cumulative": len(segments),
            })

        # ── Batch complete: dedup + progress ─────────────────────────────
        # Remove ANY consecutive same-speaker same-text duplicate.
        # (Pyannote sometimes creates adjacent turns for the same speaker.)
        deduped: List[Dict] = []
        run_count = 0
        for seg in segments:
            if (deduped
                    and deduped[-1]["speaker"] == seg["speaker"]
                    and _jaccard_words(deduped[-1]["text"], seg["text"]) > 0.85):
                run_count += 1
                if run_count >= 1:   # remove from the FIRST duplicate onwards
                    hallucinations_removed += 1
                    continue
            else:
                run_count = 0
            deduped.append(seg)
        segments = deduped

        pct = 15 + int((batch_start / max(n, 1)) * 65)
        _progress(ctx, "transcribing", pct,
                  f"Partia {batch_start//batch_size + 1}/{(n-1)//batch_size + 1}: "
                  f"{len(segments)} segm. zachowanych…")

    if not segments:
        return {"error": "No speech found after filtering. Try lower min_confidence or check audio."}

    ctx.segments = segments
    ctx.asr_engine = "whisper_diarize_first"

    overlapping_count = sum(1 for s in segments if s.get("overlapping"))

    _send(ctx, {
        "type": "diarization_update",
        "segments": _build_segments_out(ctx),
        "speaker_count": len(sorted_sp),
        "message": f"Diaryzacja-first: {len(sorted_sp)} mówców, {len(segments)} segmentów",
    })

    return {
        "segment_count": len(segments),
        "speaker_count": len(sorted_sp),
        "total_turns_from_pyannote": len(raw_turns),
        "turns_with_speech": len(segments),
        "turns_empty": len(raw_turns) - len(segments) - hallucinations_removed,
        "hallucinations_removed": hallucinations_removed,
        "multi_pass_overrides": multi_pass_overrides,
        "overlapping_segments": overlapping_count,
        "method": "diarize_first_multipass",
        "recommendation": (
            f"Pyannote gave {len(raw_turns)} turns; "
            f"{len(segments)} kept, {hallucinations_removed} hallucinations removed. "
            f"{overlapping_count} overlapping segments annotated. "
            "Run verify_transcript_quality, profile_speakers, identify_speakers next."
        ),
    }


def _tool_detect_noise_regions(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Scan full audio for silence and noise regions using energy analysis.
    Results stored in ctx.noise_regions for use by other tools.
    """
    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    from ..transcriber import find_noise_regions

    min_silence = float(args.get("min_silence_sec") or 1.5)
    energy_threshold = float(args.get("energy_threshold") or 0.005)

    _progress(ctx, "transcribing", 3, "Wykrywanie ciszy i szumu…")
    regions = find_noise_regions(
        ctx.audio_pcm, ctx.sample_rate,
        noise_threshold=energy_threshold,
        min_noise_sec=min_silence,
    )
    ctx.noise_regions = regions

    total_silence = sum(r["duration_sec"] for r in regions if r["type"] == "silence")
    return {
        "regions_found": len(regions),
        "total_silence_sec": round(total_silence, 1),
        "silence_ratio": round(total_silence / max(ctx.duration, 1), 3),
        "regions": regions[:20],  # cap output length
        "recommendation": (
            "Consider setting vad_filter_threshold lower or skipping leading silence"
            if total_silence > ctx.duration * 0.2 else "Audio quality looks fine"
        ),
    }


def _tool_tag_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Auto-tag transcript segments with semantic/quality labels.

    Tags applied:
    - 'interjection': single-word affirm/deny ("tak", "nie", "mhm", "aha", "ok")
    - 'question': text ends with '?'
    - 'low-conf': avg word probability < 0.55
    - 'long': segment duration > 20 s
    - 'silence-gap': preceded by >3 s gap (potential speaker switch)
    - 'ipa': segment contains at least one word with IPA annotation

    Also identifies interjection segments that might belong to a different speaker
    than currently assigned (flagged as 'check-speaker').
    """
    if not ctx.segments:
        return {"error": "No segments. Run transcribe_audio first."}

    INTERJECTIONS = {
        "tak", "nie", "mhm", "aha", "ok", "okay", "no", "hm", "hmm",
        "eh", "ach", "och", "oj", "uff", "he", "ha", "e", "ee",
        "dobrze", "jasne", "rozumiem", "oczywiście", "właśnie",
        "yes", "no", "yeah", "right", "sure", "uh", "um", "hmm",
    }

    import re
    tagged = 0
    interjection_count = 0
    ipa_count = 0

    for i, seg in enumerate(ctx.segments):
        seg_id = seg.get("id", i)
        tags: List[str] = []
        text = (seg.get("text_pl") or seg.get("text") or "").strip()
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)

        # Interjection detection
        clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
        words_only = clean.split()
        if len(words_only) == 1 and words_only[0] in INTERJECTIONS:
            tags.append("interjection")
            interjection_count += 1
            # Flag for speaker check if surrounded by different speaker
            prev_sp = ctx.segments[i - 1].get("speaker") if i > 0 else None
            next_sp = ctx.segments[i + 1].get("speaker") if i + 1 < len(ctx.segments) else None
            curr_sp = seg.get("speaker")
            if (prev_sp and next_sp and prev_sp == next_sp and prev_sp != curr_sp):
                tags.append("check-speaker")

        # Question
        if text.endswith("?"):
            tags.append("question")

        # Low confidence
        words_list = seg.get("words") or []
        if words_list:
            probs = [float(w.get("probability", 1.0)) for w in words_list]
            if probs and sum(probs) / len(probs) < 0.55:
                tags.append("low-conf")

        # Long segment
        if dur > 20.0:
            tags.append("long")

        # Silence gap before
        if i > 0:
            gap = (seg.get("start") or 0) - (ctx.segments[i - 1].get("end") or 0)
            if gap > 3.0:
                tags.append("silence-gap")

        # IPA annotations present
        if any(w.get("ipa") for w in (seg.get("words") or [])):
            tags.append("ipa")
            ipa_count += 1

        if tags:
            ctx.segment_tags[seg_id] = tags
            seg["tags"] = tags
            tagged += 1

    check_speaker_ids = [
        seg.get("id", i) for i, seg in enumerate(ctx.segments)
        if "check-speaker" in ctx.segment_tags.get(seg.get("id", i), [])
    ]

    return {
        "segments_tagged": tagged,
        "interjections": interjection_count,
        "ipa_segments": ipa_count,
        "check_speaker_ids": check_speaker_ids[:20],
        "tag_summary": {
            tag: sum(1 for tags in ctx.segment_tags.values() if tag in tags)
            for tag in ("interjection", "question", "low-conf", "long", "silence-gap", "ipa", "check-speaker")
        },
        "recommendation": (
            f"Call refine_speaker_assignments for {len(check_speaker_ids)} "
            "interjection segment(s) flagged as possible wrong speaker"
            if check_speaker_ids else "Speaker assignments look consistent"
        ),
    }


def _tool_merge_duplicate_speakers(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Detect and merge speaker labels that are the same physical person.

    Uses MFCC+delta cosine similarity as primary signal, with two safety guards
    that PREVENT incorrect merges:
    1. Gender guard: if both speakers have gender profiles and they differ → skip merge.
    2. Age guard: if age estimates differ by > 15 years → skip merge.
    3. Temporal overlap guard: if speakers have overlapping speech (simultaneous
       segments) → they cannot be the same person.

    Default threshold 0.99 (MFCC cosine) — merging speaker labels is destructive,
    so only near-identical voices are merged. Codec/phone compression can push two
    genuinely different voices to ~0.98 similarity, so anything below 0.99 is not a
    safe "same person" signal. Lower it only with explicit evidence of pyannote
    over-splitting one speaker (e.g. long pauses splitting a monologue).
    """
    if not ctx.segments or ctx.audio_pcm is None:
        return {"error": "Segments and audio required. Run transcribe_audio and diarize_audio first."}

    # Conservative fixed default. The principled "all-pairs-too-similar → skip"
    # guard below handles compressed/phone audio where MFCC can't separate voices,
    # so no per-recording threshold tuning is needed.
    threshold = float(args.get("similarity_threshold") or 0.99)
    # Safety floor: merging speakers is destructive and irreversible. Below 0.99
    # MFCC cosine is not a reliable "same person" signal (codec compression alone
    # pushes distinct voices to ~0.98), so clamp to avoid collapsing real speakers.
    if threshold < 0.99:
        logger.info("merge_duplicate_speakers: clamping threshold %.3f → 0.99 (safety floor)", threshold)
        threshold = 0.99

    min_dur_sec = float(args.get("min_duration_sec") or 5.0)  # need at least 5s for reliability
    sr = ctx.sample_rate

    # Collect audio chunks per speaker
    speaker_audio: Dict[str, List[np.ndarray]] = {}
    speaker_time: Dict[str, float] = {}
    # Track (start, end) for overlap detection
    speaker_intervals: Dict[str, List[tuple]] = {}

    for seg in ctx.segments:
        sp = seg.get("speaker", "?")
        start_s = seg.get("start") or 0
        end_s = seg.get("end") or 0
        start_smp = int(start_s * sr)
        end_smp   = min(int(end_s * sr), len(ctx.audio_pcm))
        chunk = ctx.audio_pcm[start_smp:end_smp]
        dur = end_s - start_s
        if len(chunk) > 0:
            speaker_audio.setdefault(sp, []).append(chunk)
            speaker_time[sp] = speaker_time.get(sp, 0.0) + dur
            speaker_intervals.setdefault(sp, []).append((start_s, end_s))

    speakers = [sp for sp, segs in speaker_audio.items()
                if sum(len(c) for c in segs) / sr >= min_dur_sec]

    if len(speakers) < 2:
        return {"skipped": True, "reason": f"Need ≥2 speakers with ≥{min_dur_sec}s audio for comparison"}

    # ── Safety guard: temporal overlap ────────────────────────────────────
    def _speakers_overlap(sp_a: str, sp_b: str, tol: float = 0.2) -> bool:
        """Return True if the two speakers have any overlapping speech."""
        for a_start, a_end in speaker_intervals.get(sp_a, []):
            for b_start, b_end in speaker_intervals.get(sp_b, []):
                overlap = min(a_end, b_end) - max(a_start, b_start)
                if overlap > tol:
                    return True
        return False

    # ── Safety guard: profile mismatch ─────────────────────────────────────
    def _profiles_conflict(sp_a: str, sp_b: str) -> Optional[str]:
        pa = ctx.speaker_profiles_raw.get(sp_a, {})
        pb = ctx.speaker_profiles_raw.get(sp_b, {})
        ga, gb = pa.get("gender"), pb.get("gender")
        # Different gender (both confident) → cannot be same person
        if ga and gb and ga != gb:
            conf_a = float(pa.get("confidence") or 0)
            conf_b = float(pb.get("confidence") or 0)
            if conf_a > 0.7 and conf_b > 0.7:
                return f"gender mismatch ({ga} vs {gb})"
        # Age difference > 15 years → unlikely same person
        aa, ab = pa.get("age_estimate"), pb.get("age_estimate")
        if aa and ab and abs(aa - ab) > 15:
            return f"age gap ({aa:.0f} vs {ab:.0f})"
        return None

    # Extract MFCC feature vectors
    def _mfcc_features(segments: List[np.ndarray]) -> Optional[np.ndarray]:
        try:
            import librosa
        except ImportError:
            return None
        combined = np.concatenate(segments)[: sr * 90]  # max 90 s for better accuracy
        if len(combined) < sr * min_dur_sec:
            return None
        try:
            mfcc = librosa.feature.mfcc(y=combined.astype(np.float32), sr=sr, n_mfcc=40)
            delta1 = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            return np.concatenate([
                np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                np.mean(delta1, axis=1), np.std(delta1, axis=1),
                np.mean(delta2, axis=1), np.std(delta2, axis=1),
            ])
        except Exception as exc:
            logger.debug("MFCC extraction failed for speaker: %s", exc)
            return None

    speaker_feat: Dict[str, np.ndarray] = {}
    for sp in speakers:
        feat = _mfcc_features(speaker_audio[sp])
        if feat is not None:
            speaker_feat[sp] = feat

    if len(speaker_feat) < 2:
        return {"skipped": True, "reason": "Insufficient audio for MFCC comparison"}

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    sp_list = list(speaker_feat.keys())
    merges: List[Dict] = []
    skipped_merges: List[Dict] = []
    merge_map: Dict[str, str] = {}

    similarity_matrix: List[Dict] = []
    for i in range(len(sp_list)):
        for j in range(i + 1, len(sp_list)):
            sim = _cosine(speaker_feat[sp_list[i]], speaker_feat[sp_list[j]])
            row = {"a": sp_list[i], "b": sp_list[j], "similarity": round(sim, 4)}
            similarity_matrix.append(row)

    # ── Anti-merge guard ──────────────────────────────────────────────────────
    # If ALL pairwise similarities exceed 0.99, MFCC features are not
    # speaker-discriminative for this recording (telephone compression, narrow-band
    # audio, or reverberant environment flatten voice characteristics).
    # Merging in this case would collapse all speakers into one — skip entirely.
    if similarity_matrix:
        min_sim = min(r["similarity"] for r in similarity_matrix)
        if min_sim > 0.99:
            logger.warning(
                "merge_duplicate_speakers: all pairs have similarity > 0.99 "
                "(min=%.4f) — MFCC features not reliable for this audio. Skipping merge.",
                min_sim,
            )
            return {
                "merged_pairs": [],
                "skipped_merges": [],
                "similarity_matrix": similarity_matrix[:20],
                "threshold": threshold,
                "message": (
                    f"Skipped — all pairwise similarities > 0.99 (min={min_sim:.4f}). "
                    "MFCC features not discriminative enough for this audio "
                    "(phone call / narrow-band / reverberant). "
                    "Speaker assignment preserved as-is."
                ),
            }

    for i in range(len(sp_list)):
        for j in range(i + 1, len(sp_list)):
            row = next((r for r in similarity_matrix if r["a"] == sp_list[i] and r["b"] == sp_list[j]), None)
            if row is None:
                continue
            sim = row["similarity"]
            if sim < threshold:
                continue

            # Safety guards before merging
            if _speakers_overlap(sp_list[i], sp_list[j]):
                skipped_merges.append({**row, "reason": "simultaneous speech (different persons)"})
                continue
            conflict = _profiles_conflict(sp_list[i], sp_list[j])
            if conflict:
                skipped_merges.append({**row, "reason": f"profile conflict: {conflict}"})
                continue

            # Safe to merge — keep speaker with more speech time
            if speaker_time.get(sp_list[i], 0) >= speaker_time.get(sp_list[j], 0):
                keep, remove = sp_list[i], sp_list[j]
            else:
                keep, remove = sp_list[j], sp_list[i]
            merges.append({"keep": keep, "remove": remove, "similarity": round(sim, 4)})
            merge_map[remove] = keep

    if not merges:
        return {
            "merged_pairs": [],
            "skipped_merges": skipped_merges,
            "similarity_matrix": similarity_matrix,
            "threshold": threshold,
            "message": "No duplicate speakers found (all candidates blocked by safety guards or below threshold)",
        }

    # Resolve transitive merges (A→B, B→C ⇒ A→C)
    def _resolve(sp: str) -> str:
        visited = set()
        while sp in merge_map and sp not in visited:
            visited.add(sp)
            sp = merge_map[sp]
        return sp

    changes = 0
    for seg in ctx.segments:
        orig = seg.get("speaker", "")
        new = _resolve(orig)
        if new != orig:
            seg["speaker"] = new
            changes += 1

    # Re-number labels by total speaking time (GŁOS_01 = most speech)
    from collections import defaultdict as _dd
    time_after: Dict[str, float] = _dd(float)
    for seg in ctx.segments:
        time_after[seg.get("speaker", "?")] += (seg.get("end") or 0) - (seg.get("start") or 0)
    sorted_speakers = sorted(time_after, key=time_after.__getitem__, reverse=True)
    renumber = {sp: f"GŁOS_{i + 1:02d}" for i, sp in enumerate(sorted_speakers)}
    for seg in ctx.segments:
        seg["speaker"] = renumber.get(seg.get("speaker", ""), seg.get("speaker", ""))

    return {
        "merged_pairs": merges,
        "skipped_merges": skipped_merges,
        "segments_changed": changes,
        "similarity_matrix": similarity_matrix[:20],
        "threshold": threshold,
        "message": (
            f"Merged {len(merges)} duplicate speaker(s); {changes} segment(s) updated. "
            + (f"{len(skipped_merges)} candidate(s) blocked by safety guards." if skipped_merges else "")
        ),
    }


# ── Tool dispatch table ───────────────────────────────────────────────────────

_TOOL_IMPL = {
    # Calibration
    "get_audio_info": _tool_get_audio_info,
    "analyze_audio_quality": _tool_analyze_audio_quality,
    "detect_noise_regions": _tool_detect_noise_regions,
    "probe_audio_fragment": _tool_probe_audio_fragment,
    "detect_speaker_count": _tool_detect_speaker_count,
    "set_transcription_params": _tool_set_transcription_params,
    # Transcription
    "transcribe_audio": _tool_transcribe_audio,
    "diarize_first_transcribe": _tool_diarize_first_transcribe,
    "verify_transcript_quality": _tool_verify_transcript_quality,
    "tag_segments": _tool_tag_segments,
    "merge_short_segments": _tool_merge_short_segments,
    "retranscribe_time_range": _tool_retranscribe_time_range,
    # Diarization
    "diarize_audio": _tool_diarize_audio,
    "refine_speaker_assignments": _tool_refine_speaker_assignments,
    "merge_duplicate_speakers": _tool_merge_duplicate_speakers,
    "profile_speakers": _tool_profile_speakers,
    "identify_speakers": _tool_identify_speakers,
    # Translation
    "translate_to_polish": _tool_translate_to_polish,
    "validate_translation_quality": _tool_validate_translation_quality,
    "retranslate_segments": _tool_retranslate_segments,
    # Analysis & synthesis
    "emit_partial_result": _tool_emit_partial_result,
    "extract_entities": _tool_extract_entities,
    "extract_quotes_and_facts": _tool_extract_quotes_and_facts,
    "detect_topics": _tool_detect_topics,
    "build_rag_index": _tool_build_rag_index,
    "summarize_transcript": _tool_summarize_transcript,
    # Memory & control
    "save_memory": _tool_save_memory,
    "finish": _tool_finish,
}


# ── Result builders ───────────────────────────────────────────────────────────

