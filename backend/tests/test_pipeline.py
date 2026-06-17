"""End-to-end orchestrator test using the GPU-free stub providers."""

import os

import pytest

os.environ.setdefault("PANDARO_ASR_BACKEND", "stub")

from pandaro.orchestrator import Orchestrator, Session  # noqa: E402
from pandaro.schemas import Phase, PhaseStatus, Preset  # noqa: E402


def _session() -> Session:
    preset = Preset(
        enabled_phases=[
            Phase.DIARIZE,
            Phase.MERGE,
            Phase.PARALINGUISTICS,
            Phase.ACOUSTICS,
            Phase.KEYWORDS,
            Phase.RAG,
        ],
        translate=False,  # avoid network (Ollama) in CI
    )
    return Session("test", "/tmp/does-not-exist.wav", preset, "nagranie.wav")


@pytest.mark.asyncio
async def test_pipeline_runs_with_stubs():
    orch = Orchestrator()
    session = _session()
    # Disable LLM-dependent phases that need Ollama by skipping summarize.
    session.analysis.preset.enabled_phases = [
        p for p in session.analysis.preset.enabled_phases
    ]

    # Run the GPU/stub phases directly (no network).
    for phase in [Phase.ASR, Phase.DIARIZE, Phase.MERGE, Phase.PARALINGUISTICS, Phase.ACOUSTICS]:
        await orch.run_phase(session, phase)

    a = session.analysis
    assert a.phases[Phase.ASR.value].status == PhaseStatus.DONE
    assert len(a.transcript.segments) == 3
    # Speakers were attributed during merge.
    assert any(seg.speaker for seg in a.transcript.segments)
    # Paralinguistics produced profiles for the diarized speakers.
    assert a.speakers and a.speakers[0].gender in {"male", "female"}
    # Acoustic fingerprint present.
    assert a.acoustics.mean_pitch_hz is not None
    # Confidence report populated, low-confidence words flagged.
    assert 0.0 < a.confidence.mean_word_confidence <= 1.0
    assert a.confidence.low_confidence_ratio > 0.0


@pytest.mark.asyncio
async def test_rag_chunks_built_without_embeddings():
    from pandaro.pipeline.rag import build_rag_chunks

    orch = Orchestrator()
    session = _session()
    await orch.run_phase(session, Phase.ASR)
    chunks = await build_rag_chunks(session.analysis.transcript, client=None, embed=False)
    assert chunks
    assert chunks[0].normalized
    assert chunks[0].phonetic
