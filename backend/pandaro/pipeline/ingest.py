"""Ingest & preprocess phase: probe media, normalize to 16 kHz mono wav.

Uses ffmpeg/ffprobe when present. In dev/CI without ffmpeg, it degrades to a
no-op that simply records the source path and a best-effort duration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("ingest")


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path: str) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def to_wav16k_mono(path: str, workdir: str | None = None) -> str:
    """Return a path to a 16 kHz mono wav, transcoding if ffmpeg is available."""
    if not have_ffmpeg():
        log.warning("ingest.no_ffmpeg", path=path)
        return path
    out_dir = Path(workdir or tempfile.mkdtemp(prefix="pandaro_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(path).stem + ".16k.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1", "-ar", "16000",
            "-af", "loudnorm",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return str(out_path)
