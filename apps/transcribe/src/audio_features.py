"""
Extensible audio feature extraction pipeline.

Each extractor is a class registered with @register_extractor.
The AudioFeatureExtractor runs all enabled extractors per speaker.

Built-in extractors
-------------------
EmotionExtractor   audeering/wav2vec2-large-robust-12-ft-emotion4
                   -> anger / happiness / neutral / sadness (per speaker)
SpeechRateExtractor  librosa-based syllable rate heuristic (no model needed)
SNRExtractor         signal-to-noise ratio estimation (no model needed)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

_EXTRACTOR_REGISTRY: Dict[str, Type["BaseExtractor"]] = {}


def register_extractor(name: str):
    """Class decorator to register an extractor under a short name."""
    def decorator(cls: Type["BaseExtractor"]) -> Type["BaseExtractor"]:
        _EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator


class BaseExtractor(ABC):
    """Base class for all audio feature extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self) -> None:
        """Load/warm-up the model. Called once at startup."""

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract features from a single audio segment (mono float32 @ sr Hz).
        Returns a flat dict of feature_name -> value.
        """


# ── Emotion extractor ─────────────────────────────────────────────────────────

@register_extractor("emotion")
class EmotionExtractor(BaseExtractor):
    """
    superb/wav2vec2-base-superb-er
    Outputs probabilities for speech emotions: angry / happy / neutral / sad / etc.
    Public, no auth required.
    """
    MODEL_ID = "superb/wav2vec2-base-superb-er"

    def __init__(self):
        self._pipe = None
        self._device = "cpu"
        self._available = False

    @property
    def name(self) -> str:
        return "emotion"

    def load(self) -> None:
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe = hf_pipeline(
                "audio-classification",
                model=self.MODEL_ID,
                device=0 if self._device == "cuda" else -1,
                top_k=None,
            )
            self._available = True
            logger.info(f"EmotionExtractor loaded ({self.MODEL_ID}) on {self._device}")
        except Exception as e:
            logger.warning(f"EmotionExtractor unavailable: {e}")

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if not self._available or self._pipe is None:
            return {}
        try:
            results = self._pipe({"array": audio.astype(np.float32), "sampling_rate": sr})
            # results: [{"label": "anger", "score": 0.3}, ...]
            probs = {r["label"]: round(float(r["score"]), 3) for r in results}
            dominant = max(probs, key=lambda k: probs[k])
            return {
                "emotion": dominant,
                "emotion_probs": probs,
            }
        except Exception as e:
            logger.warning(f"EmotionExtractor.extract failed: {e}")
            return {}


# ── Speech rate extractor ─────────────────────────────────────────────────────

@register_extractor("speech_rate")
class SpeechRateExtractor(BaseExtractor):
    """
    Heuristic speech rate estimation: syllables / second.
    Uses librosa onset envelope peaks as syllable proxy.
    No model required — always available.
    """

    @property
    def name(self) -> str:
        return "speech_rate"

    def load(self) -> None:
        pass  # no model to load

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        try:
            import librosa

            duration = len(audio) / sr
            if duration < 0.2:
                return {}

            # Use onset strength envelope as a syllable proxy
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
            # Smooth and find local maxima above threshold
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(onset_env, height=np.mean(onset_env) * 0.8, distance=4)
            syllables = len(peaks)
            rate = round(syllables / duration, 2) if duration > 0 else 0.0

            return {
                "speech_rate_syllables_per_sec": rate,
                "speech_rate_label": _rate_label(rate),
            }
        except Exception as e:
            logger.warning(f"SpeechRateExtractor.extract failed: {e}")
            return {}


def _rate_label(rate: float) -> str:
    if rate < 2.5:
        return "wolne"
    if rate < 4.5:
        return "normalne"
    return "szybkie"


# ── SNR extractor ─────────────────────────────────────────────────────────────

@register_extractor("snr")
class SNRExtractor(BaseExtractor):
    """
    Signal-to-noise ratio estimation using WADA-SNR heuristic.
    No model required — always available.
    """

    @property
    def name(self) -> str:
        return "snr"

    def load(self) -> None:
        pass

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        try:
            sig = audio.astype(np.float32)
            # RMS of whole signal
            rms_total = float(np.sqrt(np.mean(sig ** 2)))
            if rms_total < 1e-9:
                return {"snr_db": None, "snr_label": "cisza"}

            # Estimate noise floor from quietest 10% of 20 ms frames
            frame_len = int(0.02 * sr)
            frames = np.array_split(sig, max(1, len(sig) // frame_len))
            frame_rms = [float(np.sqrt(np.mean(f ** 2))) for f in frames if len(f) > 0]
            frame_rms.sort()
            noise_rms = float(np.mean(frame_rms[: max(1, len(frame_rms) // 10)]))

            if noise_rms < 1e-9:
                snr_db = 40.0
            else:
                snr_db = round(20 * float(np.log10(rms_total / noise_rms)), 1)
                snr_db = max(0.0, min(60.0, snr_db))

            return {
                "snr_db": snr_db,
                "snr_label": _snr_label(snr_db),
            }
        except Exception as e:
            logger.warning(f"SNRExtractor.extract failed: {e}")
            return {}


def _snr_label(snr: float) -> str:
    if snr < 10:
        return "niski (szum)"
    if snr < 25:
        return "sredni"
    return "wysoki (czysty)"


# ── Main coordinator ──────────────────────────────────────────────────────────

_DEFAULT_EXTRACTORS = ["emotion", "speech_rate", "snr"]


class AudioFeatureExtractor:
    """
    Loads and runs all registered extractors per speaker turn.

    Usage:
        afe = AudioFeatureExtractor()            # loads default extractors
        features = afe.extract_per_speaker(audio, sr, chunks)
        # returns: {"GLOS_01": {"emotion": "neutral", ...}, ...}
    """

    def __init__(self, enabled: Optional[List[str]] = None):
        self._extractors: Dict[str, BaseExtractor] = {}
        names = enabled if enabled is not None else _DEFAULT_EXTRACTORS

        for name in names:
            cls = _EXTRACTOR_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"Unknown extractor '{name}', skipping")
                continue
            try:
                ext = cls()
                ext.load()
                self._extractors[name] = ext
                logger.info(f"AudioFeatureExtractor: loaded '{name}'")
            except Exception as e:
                logger.warning(f"AudioFeatureExtractor: failed to load '{name}': {e}")

    @property
    def loaded_extractors(self) -> List[str]:
        return list(self._extractors.keys())

    def to_cpu(self) -> None:
        """Move all PyTorch-backed extractors to CPU."""
        import torch, gc
        for ext in self._extractors.values():
            if hasattr(ext, '_pipe') and ext._pipe is not None:
                try:
                    ext._pipe.model.to("cpu")
                    ext._device = "cpu"
                except Exception:
                    pass
        gc.collect(); torch.cuda.empty_cache()
        logger.info("AudioFeatureExtractor moved to CPU.")

    def to_gpu(self) -> None:
        """Move all PyTorch-backed extractors back to GPU."""
        import torch
        if not torch.cuda.is_available():
            return
        for ext in self._extractors.values():
            if hasattr(ext, '_pipe') and ext._pipe is not None:
                try:
                    ext._pipe.model.to("cuda")
                    ext._device = "cuda"
                except Exception:
                    pass
        logger.info("AudioFeatureExtractor moved back to GPU.")

    def extract_per_speaker(
        self,
        audio: np.ndarray,
        sr: int,
        chunks: List[Dict],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate chunks by speaker, run all extractors on each."""
        speaker_audio: Dict[str, List[np.ndarray]] = defaultdict(list)
        for chunk in chunks:
            sp = chunk.get("speaker", "GLOS_01")
            s = int(chunk["start"] * sr)
            e = min(int(chunk["end"] * sr), len(audio))
            seg = audio[s:e]
            if len(seg) > 0:
                speaker_audio[sp].append(seg)

        results: Dict[str, Dict[str, Any]] = {}
        for speaker, segments in speaker_audio.items():
            combined = np.concatenate(segments)
            combined = combined[: sr * 30]  # cap at 30 s
            speaker_feats: Dict[str, Any] = {}
            for ext in self._extractors.values():
                try:
                    feats = ext.extract(combined, sr)
                    speaker_feats.update(feats)
                except Exception as e:
                    logger.warning(f"Extractor '{ext.name}' failed for {speaker}: {e}")
            results[speaker] = speaker_feats

        return results
