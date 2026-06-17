"""Pipeline orchestrator: a re-runnable DAG of phases with streaming progress.

The orchestrator is *stateless across recordings* — it holds only an in-memory,
per-session working area (the source audio path + the evolving :class:`Analysis`)
that is discarded when the session ends. The browser owns the canonical state.

Each phase:
* checks whether it is enabled in the preset (else marks SKIPPED);
* serializes GPU usage through :data:`gpu_manager` and offloads after;
* emits progress events via an async callback for WebSocket streaming;
* can be re-run individually (``run_phase``) without redoing the whole pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from .clients import OllamaClient
from .config import Settings, get_settings
from .gpu import gpu_manager
from .logging_setup import get_logger
from .pipeline import ingest as ingest_mod
from .pipeline.merge import assign_speakers, speaker_talk_time
from .pipeline.nlp import extract_nlp
from .pipeline.rag import build_rag_chunks
from .pipeline.summarize import summarize
from .pipeline.translate import translate_transcript
from .providers import (
    build_acoustics,
    build_asr,
    build_diarization,
    build_paralinguistics,
)
from .schemas import (
    PHASE_ORDER,
    Analysis,
    Phase,
    PhaseState,
    PhaseStatus,
    Preset,
)
from .text import annotate_transcript

log = get_logger("orchestrator")

ProgressCb = Callable[[PhaseState], Awaitable[None]]


class Session:
    """In-memory working area for one recording (ephemeral)."""

    def __init__(self, session_id: str, audio_path: str, preset: Preset, filename: str | None):
        self.id = session_id
        self.audio_path = audio_path
        self.analysis = Analysis(preset=preset, media_filename=filename)
        for p in PHASE_ORDER:
            self.analysis.phases[p.value] = PhaseState(phase=p)


class Orchestrator:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.ollama = OllamaClient(self.s)

    # ------------------------------------------------------------------ #
    async def run_all(self, session: Session, on_progress: ProgressCb | None = None) -> Analysis:
        preset = session.analysis.preset
        enabled = set(preset.enabled_phases)
        for phase in PHASE_ORDER:
            if phase in (Phase.INGEST, Phase.ASR, Phase.REPORT) or phase in enabled:
                await self.run_phase(session, phase, on_progress)
            else:
                await self._set_state(session, phase, PhaseStatus.SKIPPED, 1.0, on_progress)
        return session.analysis

    async def run_phase(
        self, session: Session, phase: Phase, on_progress: ProgressCb | None = None
    ) -> Analysis:
        await self._set_state(session, phase, PhaseStatus.RUNNING, 0.0, on_progress)
        try:
            handler = getattr(self, f"_phase_{phase.value}")
            await handler(session, on_progress)
            await self._set_state(session, phase, PhaseStatus.DONE, 1.0, on_progress)
        except Exception as exc:  # noqa: BLE001
            log.exception("phase.error", phase=phase.value)
            st = session.analysis.phases[phase.value]
            st.status = PhaseStatus.ERROR
            st.error = str(exc)
            st.ended_at = time.time()
            if on_progress:
                await on_progress(st)
        return session.analysis

    async def _set_state(
        self,
        session: Session,
        phase: Phase,
        status: PhaseStatus,
        progress: float,
        on_progress: ProgressCb | None,
        message: str | None = None,
    ) -> None:
        st = session.analysis.phases[phase.value]
        st.status = status
        st.progress = progress
        st.message = message
        if status == PhaseStatus.RUNNING and st.started_at is None:
            st.started_at = time.time()
        if status in (PhaseStatus.DONE, PhaseStatus.SKIPPED, PhaseStatus.ERROR):
            st.ended_at = time.time()
        if on_progress:
            await on_progress(st)

    # ------------------------------------------------------------------ #
    # Phase handlers                                                     #
    # ------------------------------------------------------------------ #
    async def _phase_ingest(self, session: Session, _cb: ProgressCb | None) -> None:
        path = session.audio_path
        wav = ingest_mod.to_wav16k_mono(path)
        session.audio_path = wav
        dur = ingest_mod.probe_duration(wav) or session.analysis.media_duration
        session.analysis.media_duration = dur

    async def _phase_vad(self, session: Session, _cb: ProgressCb | None) -> None:
        # VAD is handled inside faster-whisper (vad_filter=True). This phase is a
        # placeholder for an explicit VAD pass / standalone Silero segmentation.
        pass

    async def _phase_asr(self, session: Session, _cb: ProgressCb | None) -> None:
        preset = session.analysis.preset
        backend = build_asr(self.s, preset)
        session.analysis.model_versions["asr"] = backend.name
        transcript = await gpu_manager.run(
            f"asr:{backend.name}", lambda: backend.transcribe(session.audio_path, preset)
        )
        if not session.analysis.media_duration:
            session.analysis.media_duration = transcript.duration
        session.analysis.transcript = transcript
        # Confidence annotation (precision is a priority).
        report = annotate_transcript(transcript, preset.confidence_threshold)
        session.analysis.confidence = report

    async def _phase_align(self, session: Session, _cb: ProgressCb | None) -> None:
        # faster-whisper already returns word timestamps; a dedicated wav2vec2
        # alignment pass would tighten them. No-op in the default stack.
        pass

    async def _phase_diarize(self, session: Session, _cb: ProgressCb | None) -> None:
        preset = session.analysis.preset
        backend = build_diarization(self.s)
        session.analysis.model_versions["diarization"] = backend.name
        turns = await gpu_manager.run(
            f"diarize:{backend.name}", lambda: backend.diarize(session.audio_path, preset)
        )
        session.analysis.diarization = turns

    async def _phase_merge(self, session: Session, _cb: ProgressCb | None) -> None:
        a = session.analysis
        assign_speakers(a.transcript, a.diarization)
        # Re-run confidence so per-speaker stats are populated after attribution.
        a.confidence = annotate_transcript(a.transcript, a.preset.confidence_threshold)

    async def _phase_speaker_id(self, session: Session, _cb: ProgressCb | None) -> None:
        # Optional: match diarized speakers against the preset voiceprint gallery.
        # Without embeddings we just map provided names positionally if given.
        hints = session.analysis.preset.speaker_hints
        if not hints:
            return
        talk = speaker_talk_time(session.analysis.diarization)
        ordered = sorted(talk, key=lambda s: -talk[s])
        for spk, hint in zip(ordered, hints, strict=False):
            for prof in session.analysis.speakers:
                if prof.speaker == spk:
                    prof.name = hint.name

    async def _phase_paralinguistics(self, session: Session, _cb: ProgressCb | None) -> None:
        backend = build_paralinguistics(self.s)
        session.analysis.model_versions["paralinguistics"] = backend.name
        profiles = await gpu_manager.run(
            f"paraling:{backend.name}",
            lambda: backend.analyze(session.audio_path, session.analysis.diarization),
        )
        session.analysis.speakers = profiles

    async def _phase_acoustics(self, session: Session, _cb: ProgressCb | None) -> None:
        backend = build_acoustics(self.s)
        session.analysis.model_versions["acoustics"] = backend.name
        feats = await gpu_manager.run(
            f"acoustics:{backend.name}", lambda: backend.features(session.audio_path)
        )
        session.analysis.acoustics = feats

    async def _phase_translate(self, session: Session, _cb: ProgressCb | None) -> None:
        preset = session.analysis.preset
        if not preset.translate:
            return
        await translate_transcript(
            session.analysis.transcript,
            target=preset.translate_target,
            client=self.ollama,
            source_languages=None,
        )

    async def _phase_keywords(self, session: Session, _cb: ProgressCb | None) -> None:
        text = session.analysis.transcript.text
        entities, keywords = await extract_nlp(text, client=self.ollama, use_llm=True)
        session.analysis.entities = entities
        session.analysis.keywords = keywords

    async def _phase_summarize(self, session: Session, _cb: ProgressCb | None) -> None:
        preset = session.analysis.preset
        session.analysis.summary = await summarize(
            session.analysis.transcript,
            client=self.ollama,
            style=preset.summary_style,
        )

    async def _phase_rag(self, session: Session, _cb: ProgressCb | None) -> None:
        chunks = await build_rag_chunks(
            session.analysis.transcript, client=self.ollama, embed=True
        )
        session.analysis.rag_chunks = chunks

    async def _phase_report(self, session: Session, _cb: ProgressCb | None) -> None:
        # Report is assembled on demand from the Analysis (see pipeline.export).
        pass
