"""Preload HuggingFace models into the image/cache at build time.

Run during ``docker build`` with ``HF_TOKEN`` provided as a *build secret* (so the
token is never baked into an image layer). Downloads gated and non-gated model
snapshots into ``HF_HOME`` which is a BuildKit cache mount + runtime volume, so
models are fetched once and reused across builds and container restarts.

The list mirrors the providers in ``pandaro.providers`` and ``pandaro.config``.
Failures are non-fatal: a model that cannot be fetched at build time (e.g. a
gated repo the token lacks access to) is logged and downloaded lazily at
runtime instead, so the build never hard-fails on a single model.
"""

from __future__ import annotations

import os
import sys

# (repo_id, kind) — kind is informational only.
MODELS: list[tuple[str, str]] = [
    # ASR (faster-whisper / CTranslate2 conversion of Whisper large-v3).
    ("Systran/faster-whisper-large-v3", "asr"),
    # WhisperX forced-alignment model for Polish (wav2vec2).
    ("jonatasgrosman/wav2vec2-large-xlsr-53-polish", "align"),
    # Diarization (gated — needs HF_TOKEN with accepted license).
    ("pyannote/speaker-diarization-3.1", "diarization"),
    ("pyannote/segmentation-3.0", "diarization"),
    # Paralinguistics: age/gender + dimensional emotion.
    ("audeering/wav2vec2-large-robust-24-ft-age-gender", "paralinguistics"),
    ("audeering/wav2vec2-large-robust-24-ft-emotion-msp-dim", "paralinguistics"),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("[preload] HF_TOKEN not set; gated models will be skipped.", flush=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - build-time only
        print(f"[preload] huggingface_hub unavailable: {exc}", flush=True)
        return 0

    failures = 0
    for repo_id, kind in MODELS:
        try:
            print(f"[preload] {kind}: {repo_id} …", flush=True)
            snapshot_download(repo_id=repo_id, token=token, resume_download=True)
            print(f"[preload]   ok: {repo_id}", flush=True)
        except Exception as exc:  # pragma: no cover - network/gating dependent
            failures += 1
            print(f"[preload]   skipped {repo_id}: {exc}", flush=True)

    print(f"[preload] done ({failures} skipped, will lazy-load at runtime).", flush=True)
    # Never fail the build on model download issues.
    return 0


if __name__ == "__main__":
    sys.exit(main())
