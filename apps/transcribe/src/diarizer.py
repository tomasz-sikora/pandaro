"""
Speaker diarization using pyannote/speaker-diarization-3.1.

Requires HF_TOKEN env var. Accept model terms at:
https://hf.co/pyannote/speaker-diarization-3.1

If pyannote fails to load (missing token, network error, etc.), a simple
energy-based heuristic is used as a last resort so transcription can still
proceed.
"""
import os
import logging
import tempfile
from typing import List, Dict, Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")


class Diarizer:
    def __init__(self):
        self._pipeline = None
        self._method = "none"
        self._init()

    # ------------------------------------------------------------------
    def _init(self):
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if not HF_TOKEN:
            logger.warning(
                "HF_TOKEN not set — pyannote/speaker-diarization-3.1 requires a "
                "HuggingFace token. Set HF_TOKEN env var and restart. "
                "Falling back to energy heuristic for speaker assignment."
            )
            self._method = "none"
            return

        try:
            self._init_pyannote()
        except Exception as e:
            logger.error(
                f"pyannote init failed: {e}. "
                "Ensure HF_TOKEN is valid and you accepted the model terms at "
                "https://hf.co/pyannote/speaker-diarization-3.1. "
                "Falling back to energy heuristic."
            )
            self._method = "none"

    def _init_pyannote(self):
        from pyannote.audio import Pipeline
        import torch

        logger.info("Loading pyannote/speaker-diarization-3.1…")
        loaded = False
        for kwargs in [{"token": HF_TOKEN}, {"use_auth_token": HF_TOKEN}]:
            try:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", **kwargs
                )
                loaded = True
                break
            except TypeError:
                continue

        if not loaded:
            raise RuntimeError("Could not load pyannote pipeline (API mismatch)")

        self._pipeline.to(torch.device(self._device))
        self._method = "pyannote"
        logger.info("pyannote/speaker-diarization-3.1 loaded.")

    def to_cpu(self) -> None:
        if self._pipeline is not None:
            try:
                import torch
                self._pipeline.to(torch.device("cpu"))
            except Exception:
                pass
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Diarizer moved to CPU.")

    def to_gpu(self) -> None:
        if self._pipeline is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))
                    logger.info("Diarizer moved back to GPU.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assigns 'speaker' field to each chunk. Returns annotated chunks."""
        if not chunks:
            return chunks

        if self._method == "pyannote":
            return self._diarize_pyannote(audio, sr, chunks)

        logger.warning(
            "pyannote not available — using energy heuristic for speaker assignment."
        )
        return self._diarize_energy(audio, sr, chunks)

    # ------------------------------------------------------------------
    def _diarize_pyannote(self, audio: np.ndarray, sr: int, chunks: List[Dict[str, Any]]):
        """
        Assign speakers using overlap-coverage — for each segment, pick the
        speaker whose turns cover the most of that segment's duration.
        This is significantly more accurate than midpoint assignment near
        speaker boundaries and for short segments.
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            tmp_path = f.name

        try:
            result = self._pipeline(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # pyannote >= 4.x returns DiarizeOutput; older returns Annotation directly
        from pyannote.core import Annotation
        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        elif isinstance(result, Annotation):
            annotation = result
        else:
            annotation = result

        # Build a list of (start, end, raw_speaker_id) turns
        turns: List[tuple] = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

        speaker_map: Dict[str, str] = {}
        counter = 0

        for chunk in chunks:
            seg_start = chunk["start"]
            seg_end = chunk["end"]
            seg_dur = max(seg_end - seg_start, 1e-6)

            # Accumulate overlap seconds per raw speaker id
            coverage: Dict[str, float] = {}
            for t_start, t_end, sp in turns:
                overlap = min(seg_end, t_end) - max(seg_start, t_start)
                if overlap > 0:
                    coverage[sp] = coverage.get(sp, 0.0) + overlap

            if coverage:
                # Pick the speaker with the most overlap
                best_raw = max(coverage, key=lambda k: coverage[k])
                # Only assign if coverage ≥ 5% of segment duration (avoid noise)
                if coverage[best_raw] / seg_dur >= 0.05:
                    if best_raw not in speaker_map:
                        counter += 1
                        speaker_map[best_raw] = f"GŁOS_{counter:02d}"
                    chunk["speaker"] = speaker_map[best_raw]
                    continue

            # Fallback: no turn covers this segment (silence gap)
            chunk["speaker"] = "GŁOS_01"

        n_speakers = len(speaker_map)
        logger.info(f"pyannote diarization: {n_speakers} speaker(s) detected")
        return chunks

    # ------------------------------------------------------------------
    def _diarize_energy(self, audio: np.ndarray, sr: int, chunks: List[Dict[str, Any]]):
        """Simple pause-based heuristic — last-resort fallback."""
        PAUSE_THRESH = 1.5  # seconds
        speaker_idx = 0
        for i, chunk in enumerate(chunks):
            if i > 0:
                gap = chunk["start"] - chunks[i - 1]["end"]
                if gap >= PAUSE_THRESH:
                    speaker_idx = (speaker_idx + 1) % 6
            chunk["speaker"] = f"GŁOS_{speaker_idx + 1:02d}"
        return chunks
