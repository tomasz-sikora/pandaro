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
from typing import Any, Dict, List, Optional

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

    def count_speakers(self, audio: np.ndarray, sr: int) -> int:
        """
        Count unique speakers directly from pyannote annotation — no chunk assignment.

        This is the correct way to estimate speaker count; calling diarize() with a
        single dummy chunk always returns 1 because the whole-fragment chunk overlaps
        with every speaker turn and gets assigned the dominant speaker only.
        """
        if self._method != "pyannote" or self._pipeline is None:
            logger.warning("pyannote not available for speaker counting — returning 2")
            return 2

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

        from pyannote.core import Annotation
        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        elif isinstance(result, Annotation):
            annotation = result
        else:
            annotation = result

        speakers = {speaker for _, _, speaker in annotation.itertracks(yield_label=True)}
        n = max(len(speakers), 1)
        logger.info("pyannote counted %d unique speaker(s) in %.1fs fragment", n, len(audio) / sr)
        return n

    # ------------------------------------------------------------------
    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        chunks: List[Dict[str, Any]],
        num_speakers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Assigns 'speaker' field to each chunk. Returns annotated chunks."""
        if not chunks:
            return chunks

        if self._method == "pyannote":
            return self._diarize_pyannote(audio, sr, chunks, num_speakers=num_speakers)

        logger.warning(
            "pyannote not available — using energy heuristic for speaker assignment."
        )
        return self._diarize_energy(audio, sr, chunks)

    # ------------------------------------------------------------------
    def _split_at_boundaries(
        self,
        chunks: List[Dict[str, Any]],
        turns: List[tuple],
    ) -> List[Dict[str, Any]]:
        """
        Split ASR chunks at pyannote speaker-turn boundaries.

        A single Whisper segment spanning two speakers is broken at the midpoint
        of the silence gap between the two pyannote turns.  When the chunk has
        word-level timestamps (``words`` field), each sub-segment gets only the
        words that fall within its time window; otherwise the full text stays
        on the first sub-segment and later pieces are empty.

        Slivers shorter than MIN_SLIVER_SEC (100 ms) are silently discarded.
        """
        MIN_SLIVER_SEC = 0.1

        if len(turns) < 2:
            return chunks

        # Cut point = midpoint of the gap between consecutive turns
        split_times = sorted(
            (turns[i][1] + turns[i + 1][0]) / 2.0
            for i in range(len(turns) - 1)
        )

        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            c_start, c_end = chunk["start"], chunk["end"]
            inner = [t for t in split_times if c_start < t < c_end]

            if not inner:
                result.append(chunk)
                continue

            words: list = chunk.get("words") or []
            split_points = [c_start] + inner + [c_end]

            for j in range(len(split_points) - 1):
                seg_start = split_points[j]
                seg_end = split_points[j + 1]

                if seg_end - seg_start < MIN_SLIVER_SEC:
                    continue

                sub_words = [
                    w for w in words
                    if seg_start <= w.get("start", 0.0) < seg_end
                ]

                if sub_words:
                    text = " ".join(w.get("word", "").strip() for w in sub_words).strip()
                elif j == 0:
                    text = chunk["text"]  # fallback: keep full text on first piece
                else:
                    text = ""

                if not text.strip() and j > 0:
                    continue  # drop empty trailing pieces

                result.append({
                    **chunk,
                    "start": seg_start,
                    "end": seg_end,
                    "text": text,
                    "words": sub_words,
                })

        return result

    # ------------------------------------------------------------------
    def _diarize_pyannote(self, audio: np.ndarray, sr: int, chunks: List[Dict[str, Any]], num_speakers: Optional[int] = None):
        """
        Four-pass speaker assignment:

        1. Split ASR chunks at pyannote turn boundaries.
        2. Assign each sub-segment to the speaker whose turns cover the most
           of its duration. Silence gaps carry forward the last known speaker.
        3. Re-number speaker labels by total speaking time (dominant → GŁOS_01).
        4. Short-segment cleanup: re-assign segments < 1.0s using majority vote
           from 5 nearest neighbours to fix mis-assigned whispers/interjections.
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            tmp_path = f.name

        try:
            diarize_kwargs = {}
            if num_speakers and num_speakers > 0:
                diarize_kwargs["num_speakers"] = num_speakers
            result = self._pipeline(tmp_path, **diarize_kwargs)
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

        # Pass 1 — split chunks at turn boundaries
        chunks = self._split_at_boundaries(chunks, turns)

        # Pass 2 — assign speakers via overlap-coverage
        speaker_map: Dict[str, str] = {}
        counter = 0
        last_speaker: Optional[str] = None
        speaking_time: Dict[str, float] = {}  # label → total seconds spoken

        for chunk in chunks:
            seg_start = chunk["start"]
            seg_end = chunk["end"]
            seg_dur = max(seg_end - seg_start, 1e-6)

            # Accumulate overlap seconds per raw pyannote speaker id
            coverage: Dict[str, float] = {}
            for t_start, t_end, sp in turns:
                overlap = min(seg_end, t_end) - max(seg_start, t_start)
                if overlap > 0:
                    coverage[sp] = coverage.get(sp, 0.0) + overlap

            assigned: Optional[str] = None
            if coverage:
                best_raw = max(coverage, key=lambda k: coverage[k])
                # Short segments (<0.5s) need lower threshold — single words/interjections
                # barely overlap any pyannote turn
                min_ratio = 0.01 if seg_dur < 0.5 else 0.05
                if coverage[best_raw] / seg_dur >= min_ratio:
                    if best_raw not in speaker_map:
                        counter += 1
                        speaker_map[best_raw] = f"GŁOS_{counter:02d}"
                    assigned = speaker_map[best_raw]

            # Carry forward the last real speaker for silence/low-coverage gaps
            if assigned is None:
                assigned = last_speaker if last_speaker is not None else "GŁOS_01"

            chunk["speaker"] = assigned
            last_speaker = assigned
            speaking_time[assigned] = speaking_time.get(assigned, 0.0) + seg_dur

        # Pass 3 — re-number by speaking time (most speech → GŁOS_01)
        if speaking_time:
            sorted_by_time = sorted(
                speaking_time, key=speaking_time.__getitem__, reverse=True
            )
            renumber: Dict[str, str] = {
                sp: f"GŁOS_{i + 1:02d}" for i, sp in enumerate(sorted_by_time)
            }
            for chunk in chunks:
                chunk["speaker"] = renumber.get(chunk["speaker"], chunk["speaker"])

        n_speakers = len(speaker_map)
        logger.info(
            f"pyannote diarization: {n_speakers} speaker(s) detected, "
            f"{len(chunks)} segments after boundary splitting"
        )

        # Pass 4 — Short-segment cleanup via neighbourhood majority vote
        # Segments shorter than 1.0 s are easy to mis-assign, especially
        # for brief affirmations ("tak", "nie", "mhm") or cross-talk snippets.
        SHORT_THRESH = 1.0
        WINDOW = 3  # look ±3 neighbours
        for i, chunk in enumerate(chunks):
            dur = chunk["end"] - chunk["start"]
            if dur >= SHORT_THRESH:
                continue
            start_j = max(0, i - WINDOW)
            end_j = min(len(chunks), i + WINDOW + 1)
            neighbours = [chunks[j]["speaker"] for j in range(start_j, end_j) if j != i]
            if not neighbours:
                continue
            from collections import Counter as _Counter
            top_sp, top_cnt = _Counter(neighbours).most_common(1)[0]
            # Re-assign only when there is a clear majority (≥ half of visible neighbours)
            if top_cnt >= max(2, len(neighbours) // 2):
                chunk["speaker"] = top_sp

        # Pass 5 — Empty-segment removal
        # Remove any segment with no transcribed text — these are pyannote turns
        # that overlap with silence/breath and produce phantom UI segments.
        # IMPORTANT: do NOT absorb the time-span of empty segments into the previous
        # one — that would make the next segment appear adjacent and trigger Pass 6 merge
        # across a different-speaker boundary.
        MIN_GAP = 0.25
        merged: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                # Drop the empty chunk — do NOT extend previous segment's end.
                # Keeping the gap preserves the speaker-boundary information.
                continue
            merged.append(chunk)
        chunks = merged

        # Pass 6 — Sentence-fragment merger (same speaker only)
        # Whisper sometimes splits a continuous sentence into multiple short segments
        # at VAD micro-silences. Merge consecutive same-speaker fragments that:
        #   - are close together (gap <= 0.4s)
        #   - don't exceed 20s combined duration
        #   - previous segment doesn't end with sentence-final punctuation
        # This runs AFTER speaker assignment so it never crosses speaker boundaries.
        import re as _re
        SENTENCE_END = _re.compile(r'[.!?…;]\s*$')
        MAX_FRAG_GAP = 0.1    # only merge truly back-to-back fragments (< 100ms gap)
        MAX_MERGED_DUR = 15.0

        fmerged: List[Dict[str, Any]] = []
        for chunk in chunks:
            txt = (chunk.get("text") or "").strip()
            if not fmerged:
                fmerged.append(chunk)
                continue
            prev = fmerged[-1]
            gap = chunk["start"] - prev["end"]
            prev_txt = (prev.get("text") or "").strip()
            same_sp = prev.get("speaker") == chunk.get("speaker")
            combined = chunk["end"] - prev["start"]

            if (same_sp
                    and gap <= MAX_FRAG_GAP
                    and combined <= MAX_MERGED_DUR
                    and not SENTENCE_END.search(prev_txt)):
                prev["text"] = (prev_txt + " " + txt).strip()
                prev["end"] = chunk["end"]
                prev["words"] = (prev.get("words") or []) + (chunk.get("words") or [])
            else:
                fmerged.append(chunk)
        chunks = fmerged

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
