"""
NVIDIA Nemotron 3.5 ASR Streaming transcription engine.

Model: nvidia/nemotron-3.5-asr-streaming-0.6b
Architecture: Cache-Aware FastConformer-RNNT (600 M params, ~1.2 GB float16)
Languages: 40 language-locales — Polish in "broad-coverage" tier (WER ~15 %)
Library: NVIDIA NeMo Framework

Installation (separate step — not in requirements.txt, heavy dependency):
  apt-get update && apt-get install -y libsndfile1 ffmpeg
  pip install Cython packaging
  pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]

Notes:
  - Runs in offline/batch mode using the widest right-context [56, 13] (1120 ms)
    for maximum accuracy.
  - Language-ID prompt conditioning: target_lang="pl-PL" or "auto".
  - No built-in diarization → pipeline's standard Diarizer is applied downstream.
  - VRAM: ~1.2 GB (float16) — can coexist with Whisper on most GPUs.
"""
import logging
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")

# FastConformer: 10 ms input hop × 8× subsampling → 80 ms per encoder frame.
_FRAME_STRIDE_S = 0.08

# ISO 639-1 → Nemotron locale
_LANG_TO_LOCALE: Dict[str, str] = {
    "en": "en-US",
    "pl": "pl-PL",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "ru": "ru-RU",
    "uk": "uk-UA",
    "tr": "tr-TR",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "vi": "vi-VN",
    "zh": "zh-CN",
    "sv": "sv-SE",
    "cs": "cs-CZ",
    "nb": "nb-NO",
    "da": "da-DK",
    "bg": "bg-BG",
    "fi": "fi-FI",
    "hr": "hr-HR",
    "sk": "sk-SK",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "et": "et-EE",
}

_LOCALE_TO_LANG: Dict[str, str] = {v: k for k, v in _LANG_TO_LOCALE.items()}

# Language tag emitted by the model at end of each segment: e.g. "<en-US>"
_LANG_TAG_RE = re.compile(r"<([a-z]{2,3}-[A-Z]{2})>")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_locale(language: Optional[str]) -> str:
    """Convert our language code / None to a Nemotron locale string."""
    if not language or language.lower() in ("auto", ""):
        return "auto"
    lang = language.lower().strip()
    # Already a full locale (e.g. "pl-PL")?
    if re.match(r"^[a-z]{2,3}-[A-Z]{2}$", lang):
        return lang
    return _LANG_TO_LOCALE.get(lang, "auto")


def _extract_detected_lang(text: str, fallback: Optional[str]) -> str:
    """Parse the language tag emitted by Nemotron (e.g. '<en-US>') → 'en'."""
    m = _LANG_TAG_RE.search(text)
    if m:
        locale = m.group(1)
        return _LOCALE_TO_LANG.get(locale, locale.split("-")[0].lower())
    if fallback and fallback not in ("auto", ""):
        return fallback.lower()
    return "unknown"


def _strip_lang_tags(text: str) -> str:
    """Remove language tags appended by Nemotron (e.g. '<en-US>')."""
    return _LANG_TAG_RE.sub("", text).strip()


def _build_segments(
    hyp: Any,
    clean_text: str,
    duration: float,
    frame_stride: float,
    silence_gap: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Convert NeMo word-level timestamps into pipeline segment dicts.

    Each segment: {"text": str, "start": float, "end": float, "speaker": str}

    Falls back to a single segment covering the full audio when timestamps are
    unavailable (NeMo may not expose them for all model variants).
    """
    timestep_data = getattr(hyp, "timestep", None)
    word_ts: Optional[list] = None

    if isinstance(timestep_data, dict):
        word_ts = timestep_data.get("word") or timestep_data.get("words")
    elif isinstance(timestep_data, list):
        word_ts = timestep_data

    if not word_ts:
        return [{"text": clean_text, "start": 0.0, "end": duration, "speaker": "SPEAKER_0"}]

    segments: List[Dict[str, Any]] = []
    buf_words: List[str] = []
    seg_start: Optional[float] = None
    prev_end: float = 0.0

    for entry in word_ts:
        # NeMo WordLevelTimestamps namedtuple: .word, .start_offset, .end_offset
        if hasattr(entry, "word"):
            word = entry.word
            t_start = getattr(entry, "start_offset", 0) * frame_stride
            t_end = getattr(entry, "end_offset", t_start) * frame_stride
        elif isinstance(entry, dict):
            word = entry.get("word", "")
            t_start = entry.get("start_offset", entry.get("start", 0)) * frame_stride
            t_end = entry.get("end_offset", entry.get("end", t_start)) * frame_stride
        else:
            continue

        word = _strip_lang_tags(word)
        if not word:
            continue

        if seg_start is None:
            seg_start = t_start
        elif buf_words and (t_start - prev_end) > silence_gap:
            segments.append({
                "text": " ".join(buf_words),
                "start": seg_start,
                "end": prev_end,
                "speaker": "SPEAKER_0",
            })
            buf_words = []
            seg_start = t_start

        buf_words.append(word)
        prev_end = t_end

    if buf_words:
        segments.append({
            "text": " ".join(buf_words),
            "start": seg_start or 0.0,
            "end": prev_end,
            "speaker": "SPEAKER_0",
        })

    if not segments:
        return [{"text": clean_text, "start": 0.0, "end": duration, "speaker": "SPEAKER_0"}]

    return segments


# ─────────────────────────────────────────────────────────────────────────────


class NemotronTranscriber:
    """
    Wrapper around nvidia/nemotron-3.5-asr-streaming-0.6b (NeMo framework).

    Runs in offline/batch mode with the widest context window [56, 13] (1120 ms)
    for maximum accuracy.  Language-ID prompt conditioning is supported; pass
    ``language=None`` or ``language="auto"`` for automatic language detection.

    This engine does NOT include speaker diarization — the standard Diarizer
    is applied by the pipeline after transcription.
    """

    def __init__(self):
        import torch
        import nemo.collections.asr as nemo_asr  # noqa: F401 — triggers NeMo init

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_id = NEMOTRON_MODEL

        logger.info(f"Loading Nemotron 3.5 ASR '{NEMOTRON_MODEL}' on {self.device}…")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=NEMOTRON_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Nemotron 3.5 ASR loaded.")

    # ------------------------------------------------------------------

    def unload_from_gpu(self) -> None:
        """Move model to CPU to free VRAM (e.g. before loading VibeVoice)."""
        if self.model is not None:
            logger.info("Offloading Nemotron 3.5 ASR to CPU…")
            self.model = self.model.cpu()
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("Nemotron 3.5 ASR offloaded.")

    def reload_to_gpu(self) -> None:
        """Move model back to GPU after VibeVoice use."""
        if self.model is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Reloading Nemotron 3.5 ASR to {device}…")
            self.model = self.model.to(device)
            self.device = device

    @property
    def is_on_gpu(self) -> bool:
        try:
            return next(self.model.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str],
        progress_cb: Callable[[int, str], None],
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Returns (chunks, detected_language, duration).
        Each chunk: {"text": str, "start": float, "end": float, "speaker": str}
        """
        import soundfile as sf
        import torch

        duration = len(audio) / 16_000
        target_locale = _to_locale(language)
        progress_cb(15, f"Nemotron 3.5 ASR: przygotowanie audio (lang={target_locale})…")

        # NeMo transcribe() expects a file path — write to temporary WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp_path = fh.name
        try:
            sf.write(tmp_path, audio.astype(np.float32), 16_000, subtype="PCM_16")

            progress_cb(20, f"Transkrypcja Nemotron 3.5 ASR (lang={target_locale})…")

            # override_config must be RNNTPromptTranscribeConfig — plain dicts
            # and OmegaConf structs are rejected by NeMo's type check.
            from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
                RNNTPromptTranscribeConfig,
            )
            override_cfg = RNNTPromptTranscribeConfig(
                target_lang=target_locale,
                batch_size=1,
                timestamps=True,
                verbose=False,
            )

            with torch.no_grad():
                results = self.model.transcribe(
                    [tmp_path],
                    override_config=override_cfg,
                )
        finally:
            os.unlink(tmp_path)

        progress_cb(60, "Parsowanie wyników Nemotron 3.5 ASR…")

        if not results:
            return [], language or "unknown", duration

        hyp = results[0]
        raw_text = hyp.text if hasattr(hyp, "text") else str(hyp)
        detected_lang = _extract_detected_lang(raw_text, language)
        clean_text = _strip_lang_tags(raw_text)

        progress_cb(65, f"Nemotron: wykryty język = {detected_lang}")

        chunks = _build_segments(hyp, clean_text, duration, _FRAME_STRIDE_S)
        return chunks, detected_lang, duration
