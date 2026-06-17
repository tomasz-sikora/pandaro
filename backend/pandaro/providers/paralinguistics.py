"""Paralinguistics (age/gender/emotion) and acoustic feature backends.

Real implementations use audeering wav2vec2 models + librosa/openSMILE; the
stubs produce plausible deterministic values so the dashboards render in dev.
"""

from __future__ import annotations

import hashlib

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..schemas import AcousticFeatures, SpeakerProfile, SpeakerTurn

log = get_logger("paraling")

_EMOTIONS = ["neutralny", "radość", "złość", "smutek", "strach", "zaskoczenie"]


def _seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


class StubParalinguistics:
    name = "stub"

    def available(self) -> bool:
        return True

    def analyze(self, audio_path: str, turns: list[SpeakerTurn]) -> list[SpeakerProfile]:
        speakers = sorted({t.speaker for t in turns})
        profiles: list[SpeakerProfile] = []
        for spk in speakers:
            seed = _seed(audio_path, spk)
            total = sum(t.end - t.start for t in turns if t.speaker == spk)
            profiles.append(
                SpeakerProfile(
                    speaker=spk,
                    total_speech_s=round(total, 2),
                    gender="female" if seed % 2 else "male",
                    age=round(25 + seed % 35 + (seed % 100) / 100, 1),
                    dominant_emotion=_EMOTIONS[seed % len(_EMOTIONS)],
                    valence=round((seed % 100) / 100, 3),
                    arousal=round(((seed >> 3) % 100) / 100, 3),
                    dominance=round(((seed >> 6) % 100) / 100, 3),
                )
            )
        return profiles


class AudeeringParalinguistics:  # pragma: no cover - requires GPU + models
    name = "audeering"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()

    def available(self) -> bool:
        # Implementation not yet complete — stay on stub until real model
        # loading + inference is wired up.
        return False

    def analyze(self, audio_path: str, turns: list[SpeakerTurn]) -> list[SpeakerProfile]:
        # Real impl: load audeering age/gender + emotion models, run per speaker,
        # aggregate. Left as an integration point; falls back to stub otherwise.
        raise NotImplementedError


class StubAcoustics:
    name = "stub"

    def available(self) -> bool:
        return True

    def features(self, audio_path: str) -> AcousticFeatures:
        seed = _seed(audio_path)
        return AcousticFeatures(
            snr_db=round(15 + seed % 20, 1),
            noise_floor_db=round(-60 + seed % 15, 1),
            mean_pitch_hz=round(110 + seed % 120, 1),
            pitch_std_hz=round(10 + seed % 30, 1),
            speech_rate_wps=round(2.0 + (seed % 20) / 10, 2),
            energy_rms=round((seed % 50) / 100, 3),
            jitter=round((seed % 30) / 1000, 4),
            shimmer=round((seed % 40) / 1000, 4),
            silence_ratio=round((seed % 30) / 100, 3),
            overlap_ratio=round((seed % 15) / 100, 3),
            background_tags=["cisza tła", "echo"] if seed % 2 else ["szum", "muzyka"],
        )


class LibrosaAcoustics:  # pragma: no cover - requires librosa + audio
    name = "librosa"

    def available(self) -> bool:
        try:
            import librosa  # noqa: F401

            return True
        except Exception:
            return False

    def features(self, audio_path: str) -> AcousticFeatures:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr) if y.size else np.array([0.0])
        return AcousticFeatures(
            mean_pitch_hz=round(float(np.nanmean(f0)), 1),
            pitch_std_hz=round(float(np.nanstd(f0)), 1),
            energy_rms=round(rms, 4),
            silence_ratio=round(float(np.mean(np.abs(y) < 0.005)), 3) if y.size else None,
        )


def build_paralinguistics(settings: Settings | None = None):
    s = settings or get_settings()
    backend = AudeeringParalinguistics(s)
    if backend.available():
        try:
            return backend
        except Exception:
            pass
    return StubParalinguistics()


def build_acoustics(settings: Settings | None = None):
    backend = LibrosaAcoustics()
    if backend.available():
        return backend
    return StubAcoustics()
