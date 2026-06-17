"""Provider interfaces (Protocols) for swappable model backends.

Each provider has a concrete GPU implementation (lazy-imported) and a
deterministic ``stub`` implementation so the full pipeline runs end-to-end on a
machine without a GPU or models (dev, CI, tests).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import (
    AcousticFeatures,
    Preset,
    SpeakerProfile,
    SpeakerTurn,
    Transcript,
)


@runtime_checkable
class ASRBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def transcribe(self, audio_path: str, preset: Preset) -> Transcript: ...


@runtime_checkable
class DiarizationBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def diarize(self, audio_path: str, preset: Preset) -> list[SpeakerTurn]: ...


@runtime_checkable
class ParalinguisticsBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def analyze(
        self, audio_path: str, turns: list[SpeakerTurn]
    ) -> list[SpeakerProfile]: ...


@runtime_checkable
class AcousticsBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def features(self, audio_path: str) -> AcousticFeatures: ...
