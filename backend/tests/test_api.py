"""API smoke tests using FastAPI's TestClient (stub providers, no GPU/network)."""

import io
import os

os.environ.setdefault("PANDARO_ASR_BACKEND", "stub")

from fastapi.testclient import TestClient  # noqa: E402

from pandaro.api.app import create_app  # noqa: E402
from pandaro.pipeline.export import export_bundle  # noqa: E402
from pandaro.schemas import Analysis, Segment, Transcript  # noqa: E402

client = TestClient(create_app())


def test_config_endpoint():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["llm_model"] == "gemma4"
    assert "asr" in body["phases"]


def test_create_session_and_get():
    files = {"file": ("nagranie.wav", io.BytesIO(b"RIFFfake"), "audio/wav")}
    data = {"preset": "{}"}
    r = client.post("/api/sessions", files=files, data=data)
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r2 = client.get(f"/api/sessions/{sid}")
    assert r2.status_code == 200
    assert "transcript" in r2.json()

    # Clear (ephemerality).
    r3 = client.delete(f"/api/sessions/{sid}")
    assert r3.json()["cleared"] is True
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_export_import_roundtrip():
    analysis = Analysis(
        media_filename="x.wav",
        transcript=Transcript(
            segments=[Segment(id=0, start=0, end=1, text="cześć", speaker="SPEAKER_00")]
        ),
    )
    bundle = export_bundle(analysis).encode()
    files = {"file": ("a.pandaro", io.BytesIO(bundle), "application/json")}
    r = client.post("/api/import", files=files)
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r_srt = client.get(f"/api/sessions/{sid}/export", params={"fmt": "srt"})
    assert "cześć" in r_srt.text
    assert "-->" in r_srt.text

    r_md = client.get(f"/api/sessions/{sid}/export", params={"fmt": "md"})
    assert "Transkrypt" in r_md.text


def test_invalid_phase_rerun():
    files = {"file": ("n.wav", io.BytesIO(b"data"), "audio/wav")}
    sid = client.post("/api/sessions", files=files, data={"preset": "{}"}).json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/phases/nonsense")
    assert r.status_code == 400
