"""
Transcription using faster-whisper.

Root cause of "stops at 46min" and large gaps within chunks:
  condition_on_previous_text=True causes a context-poisoning cascade.
  Once Whisper produces a garbage segment (hallucination, repetition) in
  a 30-second window, that text becomes the prompt for the NEXT window,
  which then produces more garbage. compression_ratio_threshold filters
  all of them, leaving large silent holes in the transcript.

Fix:
  • condition_on_previous_text=False  ← kills the cascade
  • initial_prompt only for the FIRST window of each chunk (cross-chunk
    continuity without poisoning subsequent windows)
  • VAD threshold lowered to 0.3 — keeps quiet phone-call audio
  • Chunk size 15 min — short enough that any edge-case still limits damage
  • _is_word_run() catches "nie, nie, nie…" hallucinations
"""
import os
import logging
from typing import List, Dict, Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
_SR = 16_000

CHUNK_MINUTES  = float(os.getenv("WHISPER_CHUNK_MINUTES", "15"))
OVERLAP_SECONDS = 20.0


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

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str],
        progress_cb: Callable[[int, str], None],
        with_alternatives: bool = True,
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Returns (chunks, detected_language, duration)."""
        if self.model is None:
            self.reload_to_gpu()

        total_duration = len(audio) / _SR
        chunk_samples  = int(CHUNK_MINUTES * 60 * _SR)
        overlap_samples = int(OVERLAP_SECONDS * _SR)

        progress_cb(15, f"Transkrypcja modelem {WHISPER_MODEL} ({total_duration/60:.0f} min)…")

        if len(audio) <= chunk_samples:
            chunks_audio = [(audio, 0.0)]
        else:
            chunks_audio = []
            pos = 0
            while pos < len(audio):
                end = min(pos + chunk_samples, len(audio))
                chunks_audio.append((audio[pos:end], pos / _SR))
                if end == len(audio):
                    break
                pos = end - overlap_samples
            logger.info(
                f"Split {total_duration/60:.0f}min audio into {len(chunks_audio)} chunks "
                f"of {CHUNK_MINUTES:.0f}min with {OVERLAP_SECONDS:.0f}s overlap"
            )

        all_chunks: List[Dict[str, Any]] = []
        detected_language = "pl"
        initial_prompt: Optional[str] = None

        for chunk_idx, (chunk_audio, offset_sec) in enumerate(chunks_audio):
            chunk_dur = len(chunk_audio) / _SR
            logger.info(
                f"Chunk {chunk_idx+1}/{len(chunks_audio)}: "
                f"{offset_sec/60:.1f}–{(offset_sec+chunk_dur)/60:.1f} min"
            )

            pct = 15 + int((offset_sec / max(total_duration, 1)) * 45)
            progress_cb(
                min(60, pct),
                f"Transkrypcja {offset_sec/60:.0f}–{(offset_sec+chunk_dur)/60:.0f} min "
                f"(chunk {chunk_idx+1}/{len(chunks_audio)})…"
            )

            kw: Dict[str, Any] = {
                "beam_size": 5,
                "best_of": 5,
                "word_timestamps": True,
                "temperature": 0.0,
                # CRITICAL: False prevents context-poisoning cascade within a chunk.
                # Each 30-s Whisper window starts fresh; a bad window cannot
                # contaminate all following windows.
                "condition_on_previous_text": False,
                # Repetition / quality filters
                "compression_ratio_threshold": 2.4,
                "no_speech_threshold": 0.6,
                # VAD — threshold 0.3 keeps quiet phone-call audio (was 0.45)
                "vad_filter": True,
                "vad_parameters": {
                    "threshold": 0.3,
                    "min_silence_duration_ms": 400,
                    "min_speech_duration_ms": 150,
                    "speech_pad_ms": 200,
                },
            }
            if language and language != "auto":
                kw["language"] = language
            # Seed only the FIRST window with cross-chunk context (no cascade risk)
            if initial_prompt:
                kw["initial_prompt"] = initial_prompt

            seg_gen, info = self.model.transcribe(chunk_audio, **kw)
            if chunk_idx == 0:
                detected_language = info.language

            dedup_cutoff = (offset_sec + OVERLAP_SECONDS / 2) if chunk_idx > 0 else -1.0

            chunk_segs: List[Dict[str, Any]] = []
            prev_text = ""
            for seg_i, seg in enumerate(seg_gen):
                text = seg.text.strip()
                if not text:
                    continue

                abs_start = float(seg.start) + offset_sec
                abs_end   = float(seg.end)   + offset_sec

                if abs_start < dedup_cutoff:
                    continue

                # Update prev_text BEFORE checking to prevent false-positive cascade
                is_rep = bool(prev_text and _is_repetition(text, prev_text))
                prev_text = text
                if is_rep:
                    logger.debug(f"Skipping repeat @ {abs_start:.1f}s: {text[:60]}")
                    continue

                if _is_word_run(text):
                    logger.debug(f"Skipping word-run @ {abs_start:.1f}s: {text[:60]}")
                    continue

                words: List[Dict[str, Any]] = []
                if seg.words:
                    for w in seg.words:
                        words.append({
                            "text": w.word,
                            "start": float(w.start) + offset_sec,
                            "end":   float(w.end)   + offset_sec,
                            "probability": float(w.probability),
                            "alternatives": [],
                        })

                chunk_segs.append({
                    "text":  text,
                    "start": abs_start,
                    "end":   abs_end,
                    "words": words,
                })

                if seg_i % 20 == 0:
                    pct2 = min(60, 15 + int((abs_start / max(total_duration, 1)) * 45))
                    progress_cb(
                        pct2,
                        f"Transkrypcja {abs_start/60:.1f}/{total_duration/60:.0f} min "
                        f"({len(all_chunks)+len(chunk_segs)} segm.)"
                    )

            initial_prompt = " ".join(s["text"] for s in chunk_segs[-3:])[:120] if chunk_segs else None

            all_chunks.extend(chunk_segs)
            logger.info(
                f"Chunk {chunk_idx+1}/{len(chunks_audio)}: "
                f"{len(chunk_segs)} segments kept, cumulative: {len(all_chunks)}"
            )

        all_chunks.sort(key=lambda s: s["start"])
        logger.info(
            f"Transcription done: {len(all_chunks)} segments, "
            f"lang={detected_language}, dur={total_duration:.1f}s"
        )
        return all_chunks, detected_language, total_duration


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
