"""Integration tests that use real audio (a short WAV clip).

These tests verify the ingest and stub-ASR pipeline with actual audio data.
The test clip is generated from a local ffmpeg call so it works without
network access; the full YouTube download validation runs locally only.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from pathlib import Path

import pytest

os.environ.setdefault("PANDARO_ASR_BACKEND", "stub")

from fastapi.testclient import TestClient  # noqa: E402

from pandaro.api.app import create_app  # noqa: E402
from pandaro.pipeline.ingest import probe_duration, to_wav16k_mono  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return minimal valid 16-bit PCM mono WAV bytes."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Silence frames
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------
class TestIngest:
    def test_probe_duration_valid_wav(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(make_wav_bytes(2.0))
        dur = probe_duration(str(wav))
        assert abs(dur - 2.0) < 0.1

    def test_probe_duration_missing_file(self, tmp_path):
        dur = probe_duration(str(tmp_path / "nonexistent.wav"))
        assert dur == 0.0

    def test_to_wav16k_mono_already_wav(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.write_bytes(make_wav_bytes(1.0))
        result = to_wav16k_mono(str(wav), workdir=str(tmp_path / "out"))
        out = Path(result)
        assert out.exists()
        assert out.suffix == ".wav"

    def test_to_wav16k_mono_is_mono_16khz(self, tmp_path):
        wav = tmp_path / "stereo.wav"
        # Create a simple stereo-ish file (still valid wav, 2ch)
        n = 16000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack(f"<{n * 2}h", *([0] * n * 2)))
        wav.write_bytes(buf.getvalue())
        result = to_wav16k_mono(str(wav), workdir=str(tmp_path / "out"))
        with wave.open(result) as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000


# ---------------------------------------------------------------------------
# API integration tests with real WAV upload
# ---------------------------------------------------------------------------
class TestAPIWithRealAudio:
    @pytest.fixture(autouse=True)
    def client(self):
        self._client = TestClient(create_app())

    def test_upload_real_wav_stub_asr(self):
        """Upload a valid WAV file; stub ASR produces 3 demo segments."""
        wav_bytes = make_wav_bytes(5.0)
        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        resp = self._client.post("/api/sessions", files=files, data={"preset": "{}"})
        assert resp.status_code == 200
        body = resp.json()
        sid = body["session_id"]
        assert sid

        # Session should be accessible
        sess = self._client.get(f"/api/sessions/{sid}")
        assert sess.status_code == 200
        analysis = sess.json()
        assert "transcript" in analysis
        assert "phases" in analysis

        # Run the pipeline (stub phases, no network)
        run = self._client.post(f"/api/sessions/{sid}/run")
        assert run.status_code == 200
        assert run.json()["status"] == "running"

        # Cleanup
        self._client.delete(f"/api/sessions/{sid}")

    def test_upload_oversized_file_rejected(self):
        """Files above max_upload_mb should return 413."""
        import os

        os.environ["PANDARO_MAX_UPLOAD_MB"] = "0"  # 0 MB limit
        # Re-create app with new limit
        from pandaro.config import get_settings

        get_settings.cache_clear()
        client = TestClient(create_app())
        wav_bytes = make_wav_bytes(1.0)  # even 1s is > 0 MB
        files = {"file": ("big.wav", io.BytesIO(wav_bytes), "audio/wav")}
        resp = client.post("/api/sessions", files=files, data={"preset": "{}"})
        assert resp.status_code == 413
        # Restore
        del os.environ["PANDARO_MAX_UPLOAD_MB"]
        get_settings.cache_clear()

    def test_upload_invalid_preset_rejected(self):
        wav_bytes = make_wav_bytes(1.0)
        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        resp = self._client.post(
            "/api/sessions", files=files, data={"preset": "not json at all!!!"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_export_formats(self):
        """SRT, VTT and Markdown exports work on a session that has been through ASR."""
        from pandaro.api.store import hub
        from pandaro.orchestrator import Orchestrator
        from pandaro.schemas import Phase

        # Create session via API
        wav_bytes = make_wav_bytes(2.0)
        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        sid = self._client.post(
            "/api/sessions", files=files, data={"preset": "{}"}
        ).json()["session_id"]

        # Run ASR so transcript is populated before export
        session = hub.get(sid)
        assert session is not None
        orch = Orchestrator()
        await orch.run_phase(session, Phase.ASR)

        for fmt, expected in [("srt", "-->"), ("vtt", "WEBVTT"), ("md", "Transkrypt")]:
            resp = self._client.get(f"/api/sessions/{sid}/export", params={"fmt": fmt})
            assert resp.status_code == 200, f"format {fmt} failed"
            assert expected in resp.text, f"{fmt}: expected '{expected}' in response"

        resp = self._client.get(f"/api/sessions/{sid}/export", params={"fmt": "pandaro"})
        assert resp.status_code == 200
        data = resp.json()
        assert "transcript" in data

        # Unknown format
        resp = self._client.get(f"/api/sessions/{sid}/export", params={"fmt": "xyz"})
        assert resp.status_code == 400

    def test_health_endpoint(self):
        resp = self._client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_config_has_all_phases(self):
        resp = self._client.get("/api/config")
        assert resp.status_code == 200
        phases = resp.json()["phases"]
        assert "asr" in phases
        assert "diarize" in phases
        assert "rag" in phases
        assert "summarize" in phases


# ---------------------------------------------------------------------------
# Ingest with actual audio file from disk (local-only validation)
# ---------------------------------------------------------------------------
TEST_CLIP = Path("/tmp/test_clip.wav")


@pytest.mark.skipif(not TEST_CLIP.exists(), reason="Test clip not available (local only)")
class TestRealAudioClip:
    """Process the 30-second clip extracted from the YouTube video."""

    def test_ingest_youtube_clip(self, tmp_path):
        """Ingest phase: convert clip to 16 kHz mono without errors."""
        out = to_wav16k_mono(str(TEST_CLIP), workdir=str(tmp_path))
        assert Path(out).exists()
        dur = probe_duration(out)
        assert 25 < dur < 35, f"Expected ~30s clip, got {dur:.1f}s"

    @pytest.mark.asyncio
    async def test_pipeline_stub_on_real_audio(self, tmp_path):
        """Run the full stub pipeline against the real audio clip."""
        import shutil

        from pandaro.api.store import hub
        from pandaro.orchestrator import Orchestrator
        from pandaro.schemas import Phase, PhaseStatus, Preset

        # Copy to temp so the orchestrator can transcode without polluting /tmp
        clip = tmp_path / "clip.wav"
        shutil.copy(str(TEST_CLIP), clip)

        preset = Preset(
            enabled_phases=[Phase.DIARIZE, Phase.MERGE, Phase.PARALINGUISTICS, Phase.ACOUSTICS],
            translate=False,
        )
        session = hub.create(clip.read_bytes(), "test_clip.wav", preset)
        orch = Orchestrator()

        for phase in [Phase.INGEST, Phase.ASR, Phase.DIARIZE, Phase.MERGE, Phase.ACOUSTICS]:
            await orch.run_phase(session, phase)

        a = session.analysis
        assert a.phases[Phase.ASR.value].status == PhaseStatus.DONE
        assert a.media_duration > 0
        assert a.transcript.segments
        assert a.acoustics is not None
        hub.clear(session.id)
