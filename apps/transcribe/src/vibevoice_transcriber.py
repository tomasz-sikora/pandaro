"""
VibeVoice-ASR transcription + diarization engine.

Microsoft VibeVoice-ASR (9B, Qwen2.5-7B backbone) processes up to 60 minutes
of audio in a single pass and jointly produces ASR, speaker diarization and
timestamps — no separate diarization step needed.

Installation note:
  The vibevoice package is installed from the official GitHub repo:
    git clone https://github.com/microsoft/VibeVoice.git
    pip install -e VibeVoice
  plus optionally flash-attn for faster inference on CUDA.

Output segment dict from processor.post_process_transcription():
  {
    "start_time": "00:00:01.234",   # HH:MM:SS.mmm string
    "end_time":   "00:00:04.567",
    "speaker_id": "SPK_0",
    "text":       "hello world",
  }
"""
import logging
import os
import re
import sys
import tempfile
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

VIBEVOICE_MODEL = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR")
_VIBEVOICE_LM = os.getenv("VIBEVOICE_LM", "Qwen/Qwen2.5-7B")

# Ensure /opt/VibeVoice is on sys.path (the pip wheel installs metadata-only)
_VV_SRC = "/opt/VibeVoice"
if _VV_SRC not in sys.path:
    sys.path.insert(0, _VV_SRC)

# Patch transformers AutoModel.register to allow re-registration with exist_ok=True
# This is necessary because VibeVoice registers custom model classes that may
# conflict when the same config class is already registered by transformers.
def _patch_transformers_register():
    try:
        from transformers.models.auto import auto_factory
        _orig = auto_factory._LazyAutoMapping.register
        def _patched(self, key, value, exist_ok=False):
            _orig(self, key, value, exist_ok=True)
        auto_factory._LazyAutoMapping.register = _patched
    except Exception:
        pass

_patch_transformers_register()


def _patch_qwen2_tokenizer():
    """Back-fill tokenization_qwen2_fast into transformers if missing (transformers 5+)."""
    try:
        from transformers.models import qwen2 as _qwen2_mod
        if not hasattr(_qwen2_mod, 'tokenization_qwen2_fast'):
            import types, sys as _sys
            shim = types.ModuleType('transformers.models.qwen2.tokenization_qwen2_fast')
            try:
                from transformers import Qwen2TokenizerFast
                shim.Qwen2TokenizerFast = Qwen2TokenizerFast
            except ImportError:
                from transformers import AutoTokenizer
                shim.Qwen2TokenizerFast = AutoTokenizer
            _sys.modules['transformers.models.qwen2.tokenization_qwen2_fast'] = shim
            setattr(_qwen2_mod, 'tokenization_qwen2_fast', shim)
    except Exception:
        pass


_patch_qwen2_tokenizer()


def _parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' or 'MM:SS.mmm' or bare seconds to float seconds."""
    if ts in ("N/A", "", None):
        return 0.0
    try:
        return float(ts)
    except (ValueError, TypeError):
        pass
    # Try HH:MM:SS.mmm
    m = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", str(ts))
    if m:
        h, mn, s = m.group(1), m.group(2), m.group(3)
        return int(h) * 3600 + int(mn) * 60 + float(s)
    # Try MM:SS.mmm
    m = re.match(r"(\d+):(\d+(?:\.\d+)?)", str(ts))
    if m:
        mn, s = m.group(1), m.group(2)
        return int(mn) * 60 + float(s)
    return 0.0


class VibeVoiceTranscriber:
    """
    Wrapper around microsoft/VibeVoice-ASR.

    Jointly handles ASR + speaker diarization + timestamping in one forward
    pass — no separate Diarizer needed when using this engine.
    """

    def __init__(self):
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Official recommendation: bfloat16 on CUDA (no INT8 — degrades acoustic encoder)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Choose attention implementation
        attn_impl = "sdpa"
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Using flash_attention_2 for VibeVoice-ASR")
            except ImportError:
                logger.info("flash-attn not installed, using sdpa")

        logger.info(f"Loading VibeVoice-ASR from '{VIBEVOICE_MODEL}' on {self.device} ({self.dtype})…")

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            VIBEVOICE_MODEL,
            language_model_pretrained_name=_VIBEVOICE_LM,
        )

        # Load without device_map, then move to device (mirrors official demo script)
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            VIBEVOICE_MODEL,
            dtype=self.dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            logger.info(f"VibeVoice-ASR model device: {first_param.device}, dtype: {first_param.dtype}")
        logger.info("VibeVoice-ASR loaded.")

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str],
        progress_cb: Callable[[int, str], None],
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Returns (chunks, detected_language, duration).
        Each chunk: {"text", "start", "end", "speaker"}
        Diarization is included — caller's Diarizer should be a no-op.
        """
        import torch

        duration = len(audio) / 16_000
        progress_cb(10, "Przygotowanie audio dla VibeVoice-ASR…")

        # VibeVoice expects (array, sample_rate) tuple
        # Pass as plain ndarray list with explicit sampling_rate
        # (tuple causes inhomogeneous-shape error; dict causes float() error)
        audio_input = audio.astype(np.float32)

        inputs = self.processor(
            audio=[audio_input],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        # Cast float tensors to model dtype (bfloat16 on CUDA)
        inputs = {
            k: v.to(self.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in inputs.items()
        }

        progress_cb(15, f"VibeVoice-ASR inference na {self.device}…")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        progress_cb(70, "Dekodowanie wyników…")

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        # Strip padding
        eos_positions = (
            generated_ids == self.processor.tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[: eos_positions[0] + 1]

        generated_text = self.processor.decode(
            generated_ids, skip_special_tokens=True
        )

        try:
            raw_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            logger.warning(f"post_process_transcription failed: {e}")
            raw_segments = []

        progress_cb(80, f"Przetworzono {len(raw_segments)} segmentów.")

        # Normalise output format
        chunks: List[Dict[str, Any]] = []
        speaker_map: Dict[str, str] = {}
        counter = 0

        for seg in raw_segments:
            raw_speaker = str(seg.get("speaker_id", "SPK_0"))
            if raw_speaker not in speaker_map:
                counter += 1
                speaker_map[raw_speaker] = f"GŁOS_{counter:02d}"

            text = seg.get("text", "").strip()
            if not text:
                continue

            chunks.append({
                "text": text,
                "start": _parse_timestamp(seg.get("start_time")),
                "end": _parse_timestamp(seg.get("end_time")),
                "speaker": speaker_map[raw_speaker],
            })

        # Detect language from first meaningful segment (VibeVoice is
        # language-agnostic; try a lightweight langdetect if available)
        detected_language = _detect_language(
            " ".join(c["text"] for c in chunks[:10])
        ) if chunks else "auto"

        logger.info(
            f"VibeVoice: {len(chunks)} segments, lang≈{detected_language}, "
            f"dur={duration:.1f}s"
        )
        return chunks, detected_language, duration


# ── Language detection fallback ──────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Best-effort language detection, gracefully falls back to 'auto'."""
    try:
        from langdetect import detect
        code = detect(text)
        # Map common codes to our supported set
        _MAP = {"pl": "pl", "en": "en", "ru": "ru", "uk": "uk", "de": "de"}
        return _MAP.get(code, code)
    except Exception:
        pass
    try:
        import langid
        code, _ = langid.classify(text)
        return code
    except Exception:
        pass
    return "auto"
