"""ASR backends: faster-whisper / WhisperX (GPU) and a deterministic stub.

The real backends produce word-level timestamps and per-word probabilities; the
stub fabricates a small, valid transcript so the rest of the pipeline (merge,
RAG, summarize, UI) can run without models. All backends share the same
:class:`Transcript` output contract.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..schemas import Preset, Segment, Transcript, Word

log = get_logger("asr")


def _initial_prompt(preset: Preset) -> str | None:
    """Bias rare proper nouns/jargon via Whisper's initial_prompt (precision)."""
    if not preset.vocabulary:
        return None
    return ", ".join(preset.vocabulary)


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._model = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except Exception:
            return False

    def _load(self):  # pragma: no cover - requires GPU + model
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.s.resolved_device
            compute = self.s.asr_compute_type if device == "cuda" else "int8"
            self._model = WhisperModel(
                self.s.asr_model, device=device, compute_type=compute
            )
        return self._model

    def transcribe(self, audio_path: str, preset: Preset) -> Transcript:  # pragma: no cover
        model = self._load()
        language = preset.expected_language or self.s.default_language
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=self.s.asr_beam_size,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=_initial_prompt(preset),
        )
        segments: list[Segment] = []
        for i, seg in enumerate(segments_iter):
            words = [
                Word(text=w.word, start=w.start, end=w.end, confidence=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    language=info.language,
                    words=words,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    compression_ratio=seg.compression_ratio,
                )
            )
        return Transcript(
            language=info.language,
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=segments,
        )


class StubASR:
    """Deterministic, GPU-free transcript for dev/CI/tests."""

    name = "stub"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()

    def available(self) -> bool:
        return True

    def transcribe(self, audio_path: str, preset: Preset) -> Transcript:
        lang = preset.expected_language or self.s.default_language
        demo = [
            ("Dzień dobry, dziękuję że dzwonisz do nas.", 0.0, 3.2),
            ("Chciałbym porozmawiać o umowie z firmą Kowalski.", 3.4, 7.1),
            ("Oczywiście, spotkajmy się w Warszawie w przyszłym tygodniu.", 7.3, 11.0),
        ]
        segments: list[Segment] = []
        for i, (text, start, end) in enumerate(demo):
            toks = text.split()
            step = (end - start) / max(1, len(toks))
            words = [
                Word(
                    text=t,
                    start=round(start + j * step, 3),
                    end=round(start + (j + 1) * step, 3),
                    confidence=0.92 if j % 5 else 0.4,  # sprinkle low-confidence
                )
                for j, t in enumerate(toks)
            ]
            segments.append(
                Segment(
                    id=i,
                    start=start,
                    end=end,
                    text=text,
                    language=lang,
                    words=words,
                    avg_logprob=-0.2,
                    no_speech_prob=0.02,
                    compression_ratio=1.3,
                )
            )
        return Transcript(language=lang, duration=demo[-1][2], segments=segments)


def build_asr(settings: Settings | None = None, preset: Preset | None = None):
    """Pick an ASR backend from preset/settings, falling back to the stub."""
    s = settings or get_settings()
    requested = (preset.asr_backend if preset else None) or s.asr_backend
    candidates = {
        "faster-whisper": FasterWhisperASR,
        "whisperx": FasterWhisperASR,  # WhisperX shares CT2 backend; stub-equivalent here
        "stub": StubASR,
    }
    cls = candidates.get(requested, FasterWhisperASR)
    backend = cls(s)
    if not backend.available():
        log.warning("asr.fallback_stub", requested=requested)
        return StubASR(s)
    return backend
