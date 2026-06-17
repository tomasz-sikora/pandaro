"""Diarization backends: pyannote (GPU, HF-gated) and a deterministic stub."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..schemas import Preset, SpeakerTurn

log = get_logger("diarize")


class PyannoteDiarization:
    name = "pyannote"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._pipeline = None

    def available(self) -> bool:
        try:
            import pyannote.audio  # noqa: F401

            return bool(self.s.hf_token)
        except Exception:
            return False

    def _load(self):  # pragma: no cover - requires GPU + HF token
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.s.diarization_model, use_auth_token=self.s.hf_token
            )
            if self.s.resolved_device == "cuda":
                self._pipeline.to(torch.device("cuda"))
        return self._pipeline

    def diarize(self, audio_path: str, preset: Preset) -> list[SpeakerTurn]:  # pragma: no cover
        pipeline = self._load()
        kwargs = {}
        if preset.expected_speakers:
            kwargs["num_speakers"] = preset.expected_speakers
        annotation = pipeline(audio_path, **kwargs)
        turns: list[SpeakerTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(speaker=str(speaker), start=float(turn.start), end=float(turn.end))
            )
        return turns


class StubDiarization:
    """Alternates two speakers across the recording for dev/CI."""

    name = "stub"

    def available(self) -> bool:
        return True

    def diarize(self, audio_path: str, preset: Preset) -> list[SpeakerTurn]:
        n = preset.expected_speakers or 2
        # Without audio we cannot know boundaries; orchestrator aligns these to
        # ASR segment times. Produce coarse alternating turns for 0..12s.
        turns: list[SpeakerTurn] = []
        span = 3.5
        t = 0.0
        i = 0
        while t < 12.0:
            turns.append(
                SpeakerTurn(
                    speaker=f"SPEAKER_{i % n:02d}",
                    start=round(t, 2),
                    end=round(t + span, 2),
                )
            )
            t += span
            i += 1
        return turns


def build_diarization(settings: Settings | None = None):
    s = settings or get_settings()
    backend: object
    if s.diarization_backend == "pyannote":
        backend = PyannoteDiarization(s)
        if backend.available():
            return backend
        log.warning("diarize.fallback_stub", requested="pyannote")
    return StubDiarization()
