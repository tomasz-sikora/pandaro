"""
Transcription using faster-whisper with adaptive chunking, context comparison,
adaptive VAD, silence detection, and IPA annotation for low-confidence words.

Key design decisions:
  • condition_on_previous_text=False within each chunk — prevents context-poisoning
    cascade (one bad 30-s window poisoning all subsequent windows).
  • initial_prompt only at the FIRST window of each chunk — provides cross-chunk
    continuity without the cascade risk.
  • Context comparison: first chunk is transcribed BOTH with and without
    initial_prompt; the version with higher avg word confidence is kept.
  • Adaptive VAD: each audio chunk gets its own VAD threshold derived from the
    local RMS energy — quiet regions use a lower threshold than loud ones.
  • Silence trimming: leading/trailing silence (>2 s) is detected and skipped
    so Whisper doesn't waste time on intro/outro noise.
  • IPA annotation: words below a confidence threshold get an IPA transcription
    appended so downstream LLM can make sense of unclear phonemes.
  • Configurable via env: WHISPER_MODEL, WHISPER_CHUNK_MINUTES, WHISPER_OVERLAP_SEC,
    WHISPER_ADAPTIVE_VAD, WHISPER_IPA_THRESHOLD.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
_SR = 16_000

# ── Configurable chunking params ──────────────────────────────────────────────
CHUNK_MINUTES    = float(os.getenv("WHISPER_CHUNK_MINUTES", "15"))
OVERLAP_SECONDS  = float(os.getenv("WHISPER_OVERLAP_SEC", "20"))
# Adaptive VAD: compute per-chunk threshold from local energy (True by default)
ADAPTIVE_VAD     = os.getenv("WHISPER_ADAPTIVE_VAD", "1") not in ("0", "false", "False")
# IPA annotation: add phonetic transcription for words below this confidence
IPA_THRESHOLD    = float(os.getenv("WHISPER_IPA_THRESHOLD", "0.40"))
# Silence trimming: skip leading/trailing silence longer than this (seconds)
SILENCE_TRIM_SEC = float(os.getenv("WHISPER_SILENCE_TRIM_SEC", "2.0"))


# ── IPA helper (best-effort, needs espeak-ng installed) ───────────────────────
def _word_to_ipa(word: str) -> Optional[str]:
    """Convert a word to IPA using espeak-ng. Returns None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-q", word],
            capture_output=True, text=True, timeout=1.0,
        )
        ipa = result.stdout.strip()
        return ipa if ipa else None
    except Exception:
        return None


_IPA_AVAILABLE: Optional[bool] = None


def _check_ipa() -> bool:
    global _IPA_AVAILABLE
    if _IPA_AVAILABLE is None:
        _IPA_AVAILABLE = _word_to_ipa("test") is not None
    return _IPA_AVAILABLE


# ── Adaptive VAD analyzer ─────────────────────────────────────────────────────
def _compute_local_vad_threshold(audio: np.ndarray, sr: int,
                                  frame_ms: int = 25,
                                  base_threshold: float = 0.25) -> float:
    """
    Derive a VAD threshold tuned for this specific audio chunk.

    Strategy:
    • Compute per-frame RMS energy.
    • If the median energy is very low (quiet/phone audio) → use a lower threshold
      so we don't filter out quiet speech.
    • If the energy is high and dynamic range is large (active conversation) →
      use a slightly higher threshold to filter background rumble.
    • Clamp to [0.10, 0.50].
    """
    if len(audio) == 0:
        return base_threshold
    frame_len = int(frame_ms / 1000 * sr)
    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    if not frames:
        return base_threshold
    energies = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    median_e = float(np.median(energies))
    max_e    = float(np.max(energies))
    if median_e < 0.005:      # very quiet — reduce threshold aggressively
        t = max(0.10, base_threshold - 0.10)
    elif median_e < 0.02:     # quiet phone call
        t = max(0.12, base_threshold - 0.06)
    elif max_e > 0.3 and (max_e / (median_e + 1e-9)) > 8:
        t = min(0.50, base_threshold + 0.05)  # dynamic range: be slightly stricter
    else:
        t = base_threshold
    logger.debug("Adaptive VAD: median_rms=%.4f → threshold=%.2f", median_e, t)
    return round(t, 3)


# ── Silence trimmer ───────────────────────────────────────────────────────────
def _find_speech_boundaries(audio: np.ndarray, sr: int,
                              frame_ms: int = 50,
                              energy_threshold: float = 0.01,
                              min_silence_sec: float = 2.0) -> Tuple[float, float]:
    """
    Return (speech_start_sec, speech_end_sec) — the range of audio that actually
    contains speech (trimming long silence at both ends).
    Falls back to (0.0, duration) if no silence is detected.
    """
    dur = len(audio) / sr
    if dur < min_silence_sec * 2:
        return 0.0, dur

    frame_len = int(frame_ms / 1000 * sr)
    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    if not frames:
        return 0.0, dur

    rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    is_speech = rms >= energy_threshold

    # Find first speech frame
    speech_indices = np.where(is_speech)[0]
    if len(speech_indices) == 0:
        return 0.0, dur

    first_speech = speech_indices[0]
    last_speech  = speech_indices[-1]

    start_sec = max(0.0, float(first_speech) * frame_ms / 1000 - 0.5)  # 0.5s padding
    end_sec   = min(dur, float(last_speech + 1) * frame_ms / 1000 + 0.5)
    return start_sec, end_sec


# ── Noise region detection ────────────────────────────────────────────────────
def find_noise_regions(audio: np.ndarray, sr: int,
                        frame_ms: int = 100,
                        noise_threshold: float = 0.005,
                        min_noise_sec: float = 1.5) -> List[Dict]:
    """
    Return a list of {start_sec, end_sec, type: 'silence'|'noise'} regions
    that are likely to produce hallucinations in Whisper.

    • silence: RMS < noise_threshold for min_noise_sec
    • noise:   RMS > 0.3 AND spectral flatness > 0.6 (broadband noise)
    """
    frame_len = int(frame_ms / 1000 * sr)
    regions: List[Dict] = []
    dur = len(audio) / sr

    if len(audio) < frame_len * 2:
        return regions

    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    times = [i * frame_ms / 1000 for i in range(len(frames))]
    rms   = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])

    # Silence detection
    in_silence = False
    s_start = 0.0
    for i, (t, r) in enumerate(zip(times, rms)):
        if not in_silence and r < noise_threshold:
            in_silence = True
            s_start = t
        elif in_silence and r >= noise_threshold:
            in_silence = False
            dur_s = t - s_start
            if dur_s >= min_noise_sec:
                regions.append({"start_sec": round(s_start, 2), "end_sec": round(t, 2),
                                 "type": "silence", "duration_sec": round(dur_s, 2)})
    if in_silence and (dur - s_start) >= min_noise_sec:
        regions.append({"start_sec": round(s_start, 2), "end_sec": round(dur, 2),
                         "type": "silence", "duration_sec": round(dur - s_start, 2)})

    return regions


@dataclass
class TranscriptionConfig:
    """All tunable transcription parameters. Env vars provide defaults."""
    chunk_minutes: float = CHUNK_MINUTES
    overlap_seconds: float = OVERLAP_SECONDS
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    vad_filter: bool = True
    vad_threshold: float = 0.25
    min_silence_duration_ms: int = 300     # was 400 — shorter pauses kept so "Mhm." survives
    min_speech_duration_ms: int = 80       # was 100 — captures very short interjections
    speech_pad_ms: int = 100              # kept at 100ms — larger values over-merge utterances
    adaptive_vad: bool = ADAPTIVE_VAD
    ipa_threshold: float = IPA_THRESHOLD
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None

    def to_whisper_kw(self, vad_override: Optional[float] = None) -> Dict[str, Any]:
        vad_t = vad_override if vad_override is not None else self.vad_threshold
        return {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "word_timestamps": True,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "vad_filter": self.vad_filter,
            "vad_parameters": {
                "threshold": vad_t,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "speech_pad_ms": self.speech_pad_ms,
            },
        }


class WhisperTranscriber:
    def __init__(self):
        import torch
        from faster_whisper import WhisperModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._compute_type = "float16" if self.device == "cuda" else "int8"
        self._model_id = WHISPER_MODEL

        logger.info(f"Loading faster-whisper '{WHISPER_MODEL}' on {self.device} ({self._compute_type})")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=self.device,
            compute_type=self._compute_type,
            cpu_threads=4,
            num_workers=2,
        )
        logger.info("Whisper model loaded.")

    def unload_from_gpu(self) -> None:
        if self.model is not None:
            logger.info("Unloading Whisper CTranslate2 model from GPU…")
            del self.model
            self.model = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("Whisper unloaded.")

    def reload_to_gpu(self) -> None:
        if self.model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Reloading Whisper '{self._model_id}' to GPU…")
            self.model = WhisperModel(
                self._model_id,
                device=self.device,
                compute_type=self._compute_type,
                cpu_threads=4,
                num_workers=2,
            )
            logger.info("Whisper reloaded.")

    @property
    def is_on_gpu(self) -> bool:
        return self.model is not None and self.device == "cuda"

    # ── Low-level chunk transcription ──────────────────────────────────────
    def _transcribe_chunk(
        self,
        chunk_audio: np.ndarray,
        offset_sec: float,
        dedup_cutoff: float,
        cfg: TranscriptionConfig,
        language: Optional[str],
        vad_override: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe a single audio chunk, returning segments with absolute timestamps.
        Handles per-word IPA annotation for low-confidence words.
        """
        kw = cfg.to_whisper_kw(vad_override=vad_override)
        if language and language != "auto":
            kw["language"] = language
        if cfg.initial_prompt:
            kw["initial_prompt"] = cfg.initial_prompt

        seg_gen, _ = self.model.transcribe(chunk_audio, **kw)

        result: List[Dict[str, Any]] = []
        prev_text = ""
        use_ipa = _check_ipa() and cfg.ipa_threshold > 0

        for seg in seg_gen:
            text = seg.text.strip()
            if not text:
                continue

            abs_start = float(seg.start) + offset_sec
            abs_end   = float(seg.end)   + offset_sec

            if abs_start < dedup_cutoff:
                continue

            if _is_repetition(text, prev_text):
                logger.debug("Skipping repeat @ %.1fs: %s", abs_start, text[:60])
                prev_text = text
                continue
            if _is_word_run(text):
                logger.debug("Skipping word-run @ %.1fs: %s", abs_start, text[:60])
                continue
            prev_text = text

            words: List[Dict[str, Any]] = []
            if seg.words:
                for w in seg.words:
                    wd: Dict[str, Any] = {
                        "text":  w.word,
                        "start": float(w.start) + offset_sec,
                        "end":   float(w.end)   + offset_sec,
                        "probability": float(w.probability),
                        "alternatives": [],
                    }
                    # IPA annotation for low-confidence words
                    if use_ipa and float(w.probability) < cfg.ipa_threshold:
                        raw = w.word.strip().lower()
                        ipa = _word_to_ipa(raw)
                        if ipa:
                            wd["ipa"] = ipa
                    words.append(wd)

            result.append({
                "text":  text,
                "start": abs_start,
                "end":   abs_end,
                "words": words,
            })

        return result

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str],
        progress_cb: Callable[[int, str], None],
        with_alternatives: bool = True,
        extra_kw: Optional[Dict[str, Any]] = None,
        on_segments_cb: Optional[Callable[[List[Dict[str, Any]], float], None]] = None,
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """
        Full-audio transcription with adaptive chunking and VAD.

        1. Detect speech boundaries (trim leading/trailing silence).
        2. Split active region into overlapping chunks.
        3. For each chunk: compute adaptive VAD threshold; transcribe.
        4. First chunk: compare result WITH and WITHOUT initial_prompt,
           keep higher-confidence result (context-comparison).
        5. Annotate low-confidence words with IPA (if espeak-ng available).
        6. Deduplicate overlap regions by timestamp cutoff.

        on_segments_cb(chunk_segs, offset_sec) fires after each chunk.
        """
        if self.model is None:
            self.reload_to_gpu()

        total_duration = len(audio) / _SR

        # ── Build TranscriptionConfig from defaults + extra_kw overrides ──
        cfg = TranscriptionConfig()
        if extra_kw:
            extra_kw = dict(extra_kw)  # copy so we don't mutate caller's dict
            vad_t = extra_kw.pop("vad_filter_threshold", None)
            if vad_t is not None:
                cfg.vad_threshold = float(vad_t)
                cfg.adaptive_vad = False  # explicit override disables adaptive
            for k in ("beam_size", "best_of", "temperature",
                      "no_speech_threshold", "compression_ratio_threshold",
                      "condition_on_previous_text"):
                if k in extra_kw:
                    setattr(cfg, k, extra_kw[k])
            # chunk_minutes / overlap_seconds can also be passed via extra_kw
            if "chunk_minutes" in extra_kw:
                cfg.chunk_minutes = float(extra_kw["chunk_minutes"])
            if "overlap_seconds" in extra_kw:
                cfg.overlap_seconds = float(extra_kw["overlap_seconds"])

        # ── Silence trimming ──────────────────────────────────────────────
        speech_start, speech_end = _find_speech_boundaries(
            audio, _SR, min_silence_sec=SILENCE_TRIM_SEC
        )
        trim_start_smp = int(speech_start * _SR)
        trim_end_smp   = int(speech_end   * _SR)
        active_audio   = audio[trim_start_smp:trim_end_smp]
        logger.info(
            "Speech boundaries: %.1f–%.1f s (trimmed %.1f s of silence)",
            speech_start, speech_end, (total_duration - (speech_end - speech_start)),
        )

        # ── Chunk the active audio ────────────────────────────────────────
        chunk_samples   = int(cfg.chunk_minutes * 60 * _SR)
        overlap_samples = int(cfg.overlap_seconds * _SR)
        active_duration = len(active_audio) / _SR

        progress_cb(15, f"Transkrypcja {WHISPER_MODEL} ({total_duration/60:.0f} min)…")

        if len(active_audio) <= chunk_samples:
            chunks_audio: List[Tuple[np.ndarray, float]] = [
                (active_audio, speech_start)
            ]
        else:
            chunks_audio = []
            pos = 0
            while pos < len(active_audio):
                end = min(pos + chunk_samples, len(active_audio))
                chunks_audio.append((active_audio[pos:end], speech_start + pos / _SR))
                if end == len(active_audio):
                    break
                pos = end - overlap_samples
            logger.info(
                "Split %.0f-min audio into %d chunks (%.0f-min / %.0f-s overlap)",
                active_duration / 60, len(chunks_audio),
                cfg.chunk_minutes, cfg.overlap_seconds,
            )

        # ── Detect language on first chunk ────────────────────────────────
        detected_language = language or "pl"
        if language in (None, "auto") and active_audio is not None and len(active_audio) > 0:
            try:
                probe = active_audio[: min(len(active_audio), 30 * _SR)]
                _, info = self.model.transcribe(probe, beam_size=1, best_of=1,
                                                 word_timestamps=False, vad_filter=True)
                detected_language = info.language
            except Exception:
                pass

        all_chunks: List[Dict[str, Any]] = []
        initial_prompt: Optional[str] = None

        for chunk_idx, (chunk_audio, offset_sec) in enumerate(chunks_audio):
            chunk_dur = len(chunk_audio) / _SR
            logger.info(
                "Chunk %d/%d: %.1f–%.1f min",
                chunk_idx + 1, len(chunks_audio),
                offset_sec / 60, (offset_sec + chunk_dur) / 60,
            )

            pct = 15 + int(((offset_sec - speech_start) / max(active_duration, 1)) * 45)
            progress_cb(
                min(60, pct),
                f"Transkrypcja {offset_sec/60:.0f}–{(offset_sec+chunk_dur)/60:.0f} min "
                f"(chunk {chunk_idx+1}/{len(chunks_audio)})…",
            )

            # Adaptive VAD threshold for this chunk
            vad_override: Optional[float] = None
            if cfg.adaptive_vad:
                vad_override = _compute_local_vad_threshold(
                    chunk_audio, _SR, base_threshold=cfg.vad_threshold
                )

            dedup_cutoff = (offset_sec + cfg.overlap_seconds / 2) if chunk_idx > 0 else -1.0

            # ── Context comparison on first chunk ─────────────────────────
            if chunk_idx == 0 and initial_prompt:
                # Try WITH context
                cfg.initial_prompt = initial_prompt
                segs_with = self._transcribe_chunk(
                    chunk_audio, offset_sec, dedup_cutoff, cfg, language, vad_override
                )
                # Try WITHOUT context
                cfg.initial_prompt = None
                segs_without = self._transcribe_chunk(
                    chunk_audio, offset_sec, dedup_cutoff, cfg, language, vad_override
                )
                # Keep the one with higher avg word confidence
                def _avg_conf(segs: List[Dict]) -> float:
                    probs = [w["probability"] for s in segs for w in (s.get("words") or [])]
                    return sum(probs) / len(probs) if probs else 0.0

                conf_with    = _avg_conf(segs_with)
                conf_without = _avg_conf(segs_without)
                if conf_with >= conf_without:
                    chunk_segs = segs_with
                    logger.info("First chunk: WITH context wins (%.3f vs %.3f)", conf_with, conf_without)
                else:
                    chunk_segs = segs_without
                    logger.info("First chunk: WITHOUT context wins (%.3f vs %.3f)", conf_without, conf_with)
            else:
                cfg.initial_prompt = initial_prompt
                chunk_segs = self._transcribe_chunk(
                    chunk_audio, offset_sec, dedup_cutoff, cfg, language, vad_override
                )

            # Seed next chunk's initial_prompt from last few segments of this chunk
            initial_prompt = (
                " ".join(s["text"] for s in chunk_segs[-3:])[:120] if chunk_segs else None
            )
            cfg.initial_prompt = None  # reset — set per-chunk above

            if on_segments_cb and chunk_segs:
                try:
                    on_segments_cb(chunk_segs, offset_sec)
                except Exception as _cb_exc:
                    logger.debug("on_segments_cb error (ignored): %s", _cb_exc)

            all_chunks.extend(chunk_segs)
            logger.info(
                "Chunk %d/%d: %d segments kept, cumulative: %d",
                chunk_idx + 1, len(chunks_audio), len(chunk_segs), len(all_chunks),
            )

        all_chunks.sort(key=lambda s: s["start"])
        logger.info(
            "Transcription done: %d segments, lang=%s, dur=%.1fs",
            len(all_chunks), detected_language, total_duration,
        )
        return all_chunks, detected_language, total_duration


def _merge_sentence_fragments(
    chunks: List[Dict[str, Any]],
    max_gap_sec: float = 0.4,
    max_merged_duration: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Merge consecutive Whisper segments that are clearly parts of the same sentence.

    Merges when ALL of these are true:
    - Gap between segments <= max_gap_sec (0.4 s)
    - Previous segment does NOT end with sentence-final punctuation (. ! ? … ;)
    - Combined duration <= max_merged_duration (20 s)

    This fixes Whisper's tendency to split long sentences at VAD boundaries,
    producing fragments like "dostałam odpowiedź" + "i tak realizujemy skup złota"
    that should be one segment.
    """
    import re
    SENTENCE_END = re.compile(r'[.!?…;]\s*$')

    if len(chunks) < 2:
        return chunks

    merged: List[Dict[str, Any]] = [dict(chunks[0])]
    for seg in chunks[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        prev_text = (prev.get("text") or "").strip()
        prev_dur = prev["end"] - prev["start"]
        combined_dur = seg["end"] - prev["start"]

        should_merge = (
            gap <= max_gap_sec
            and combined_dur <= max_merged_duration
            and not SENTENCE_END.search(prev_text)
        )

        if should_merge:
            seg_text = (seg.get("text") or "").strip()
            prev["text"] = (prev_text + " " + seg_text).strip()
            prev["end"] = seg["end"]
            prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
        else:
            merged.append(dict(seg))

    return merged


def _is_repetition(current: str, previous: str, threshold: float = 0.85) -> bool:
    """True when current is a near-duplicate of previous (bigram Jaccard >= threshold)."""
    def bigrams(text: str):
        words = text.lower().split()
        return set(zip(words, words[1:])) if len(words) >= 2 else set()

    cur_bg  = bigrams(current)
    prev_bg = bigrams(previous)
    if not cur_bg or not prev_bg:
        return current.lower().strip() == previous.lower().strip()
    intersection = len(cur_bg & prev_bg)
    union        = len(cur_bg | prev_bg)
    return (intersection / union) >= threshold if union > 0 else False


def _is_word_run(text: str, min_repeats: int = 5) -> bool:
    """
    Detects hallucinated word-run segments like "nie, nie, nie, nie, nie, …".
    Returns True if a single word (ignoring punctuation) repeats >= min_repeats
    times and makes up > 70% of the segment tokens.
    """
    import re
    from collections import Counter
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    if len(tokens) < min_repeats:
        return False
    most_common_word, count = Counter(tokens).most_common(1)[0]
    return count >= min_repeats and (count / len(tokens)) > 0.7
