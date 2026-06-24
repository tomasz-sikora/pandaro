"""
Agent-based audio processing orchestrator.

Replaces the static _pipeline() with an Ollama tool-calling loop.
gemma4:26b decides which tools to call, in what order, and with what parameters.
On errors it retries up to MAX_RETRIES_PER_TOOL times.
After each session it can save learned patterns to memory.

VRAM notes (single RTX/A100 class GPU):
  - whisper large-v3:     ~6 GB  (CTranslate2, stays on GPU)
  - vibevoice 9B:         ~18 GB (loaded lazily, unloaded after use)
  - vibevoice 9B:         ~18 GB (loaded lazily, unloaded after use)
  - diarizer (pyannote):  ~1 GB
  - profiler (wav2vec2):  ~0.5 GB
  - gemma4:26b (ollama):  external process, managed by Ollama

Agent call options:
  - num_ctx=8192  for quick tool-decision turns  (fast, low memory)
  - num_ctx=16384 for NER / summary turns        (needs more context)

Tools available:
  Probing:   probe_audio_fragment, detect_speaker_count
  Params:    set_transcription_params
  Core:      diarize_first_transcribe, transcribe_audio, diarize_audio, profile_speakers
  Post-proc: translate_to_polish, identify_speakers, merge_short_segments
  Quality:   verify_transcript_quality
  Analysis:  extract_entities, summarize_transcript, detect_topics
  Index:     build_rag_index
  Context:   emit_partial_result
  Memory:    save_memory
  Control:   finish
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

# memory module is imported in prompts.py and __init__.py only

logger = logging.getLogger(__name__)

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── Agent tuning ──────────────────────────────────────────────────────────────
MAX_STEPS = 50
MAX_RETRIES_PER_TOOL = 3
AGENT_NUM_CTX = 65536  # 64k
SUMMARY_NUM_CTX = 65536 # 64k
AGENT_TEMPERATURE = 0.05
KEEP_ALIVE = -1            # keep model loaded during session

# ── Session registry (hint injection + status) ────────────────────────────────
_active_sessions: Dict[str, "AgentContext"] = {}
_sessions_lock = threading.Lock()


def register_session(ctx: "AgentContext") -> None:
    with _sessions_lock:
        _active_sessions[ctx.session_id] = ctx


def deregister_session(session_id: str) -> None:
    with _sessions_lock:
        _active_sessions.pop(session_id, None)


def inject_hint(session_id: str, hint: str) -> bool:
    """Thread-safe hint injection into a running session."""
    with _sessions_lock:
        ctx = _active_sessions.get(session_id)
    if ctx is None:
        return False
    ctx.pending_hints.append(hint)
    logger.info("Hint injected into session %s: %s", session_id, hint[:80])
    return True


def cancel_session(session_id: str) -> bool:
    """Thread-safe cancellation request for a running session.

    Sets the cancelled flag; the agent loop checks it between steps and inside
    long-running tools (transcription batches) and aborts cleanly.
    """
    with _sessions_lock:
        ctx = _active_sessions.get(session_id)
    if ctx is None:
        return False
    ctx.cancelled = True
    logger.info("Cancellation requested for session %s", session_id)
    return True


def is_busy() -> bool:
    """True if any analysis session is currently active (single-analysis policy)."""
    with _sessions_lock:
        return len(_active_sessions) > 0


def get_active_sessions() -> List[Dict]:
    with _sessions_lock:
        return [
            {"session_id": sid, "filename": ctx.filename, "step": ctx.current_step}
            for sid, ctx in _active_sessions.items()
        ]


@dataclass
class AgentContext:
    """All mutable state shared across tool invocations in one session."""
    audio_content: bytes
    filename: str
    language_hint: Optional[str]
    do_translate: bool
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop

    # Unique session ID for checkpointing
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # LLM model for this session (overrides OLLAMA_MODEL env default)
    ollama_model: str = field(default_factory=lambda: OLLAMA_MODEL)

    def __post_init__(self) -> None:
        # If caller passed None (no override), fall back to env default
        if not self.ollama_model:
            self.ollama_model = OLLAMA_MODEL

    # State built incrementally by tools
    audio_pcm: Optional[np.ndarray] = None
    sample_rate: int = 16000
    duration: float = 0.0
    asr_engine: str = "whisper"
    segments: List[Dict] = field(default_factory=list)
    detected_language: str = "auto"
    speaker_profiles_raw: Dict = field(default_factory=dict)
    audio_features_raw: Dict = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)
    entities: Optional[Dict] = None
    rag_entries: List[Dict] = field(default_factory=list)
    summary: Optional[str] = None
    report: Optional[str] = None
    audio_sha: Optional[str] = None
    quality_stats: Optional[Dict] = None

    # Agent-tunable transcription parameters
    transcription_params: Dict = field(default_factory=dict)

    # Retry tracking per tool name
    tool_attempts: Dict[str, int] = field(default_factory=dict)

    # Whether partial result has been sent
    partial_emitted: bool = False

    # Human hints injected mid-session
    pending_hints: Deque[str] = field(default_factory=deque)

    # Cancellation flag — set by cancel_session(), checked by the agent loop
    cancelled: bool = False

    # Current processing step (for status reporting)
    current_step: int = 0

    # Topics detected
    topics: List[Dict] = field(default_factory=list)

    # Per-segment quality scores (from verify_transcript_quality)
    segment_quality: Dict[int, float] = field(default_factory=dict)

    # Detected noise/silence regions (from detect_noise_regions)
    noise_regions: List[Dict] = field(default_factory=list)

    # Segment tags: {segment_id: [tag, ...]}  (e.g. 'interjection', 'question', 'low-conf')
    segment_tags: Dict[int, List[str]] = field(default_factory=dict)

    # Context management: max segments to include in LLM tool results
    # (prevents context overflow on long recordings)
    max_ctx_segments: int = int(os.getenv("AGENT_MAX_CTX_SEGMENTS", "150"))
    # Max chars per LLM text window for entity/summary tools
    llm_chunk_chars: int = int(os.getenv("AGENT_LLM_CHUNK_CHARS", "6000"))


# ── Helpers ───────────────────────────────────────────────────────────────────

