"""
Agent-based audio processing orchestrator.

Replaces the static _pipeline() with an Ollama tool-calling loop.
gemma4:26b decides which tools to call, in what order, and with what parameters.
On errors it retries up to MAX_RETRIES_PER_TOOL times.
After each session it can save learned patterns to memory.

VRAM notes (single RTX/A100 class GPU):
  - whisper large-v3:     ~6 GB  (CTranslate2, stays on GPU)
  - vibevoice 9B:         ~18 GB (loaded lazily, unloaded after use)
  - nemotron 600M:        ~1.2 GB (stays on GPU)
  - diarizer (pyannote):  ~1 GB
  - profiler (wav2vec2):  ~0.5 GB
  - gemma4:26b (ollama):  external process, managed by Ollama

Agent call options:
  - num_ctx=8192  for quick tool-decision turns  (fast, low memory)
  - num_ctx=16384 for NER / summary turns        (needs more context)

Tools available:
  Probing:   probe_audio_fragment, detect_speaker_count
  Params:    set_transcription_params
  Core:      transcribe_audio, diarize_audio, profile_speakers
  Post-proc: translate_to_polish, identify_speakers, merge_short_segments
  Quality:   verify_transcript_quality
  Analysis:  extract_entities, summarize_transcript, run_analysis
  Index:     build_rag_index
  Context:   emit_partial_result, save_checkpoint, load_checkpoint
  Memory:    save_memory
  Control:   finish
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import threading
import uuid
from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
import numpy as np

from .memory import save_memory, format_memories_for_prompt, load_memories

logger = logging.getLogger(__name__)

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHECKPOINT_DIR = Path(os.getenv("AGENT_CHECKPOINT_DIR", "/app/data/checkpoints"))
ARTIFACTS_DIR = Path(os.getenv("AGENT_ARTIFACTS_DIR", "/app/data/artifacts"))

# ── Agent tuning ──────────────────────────────────────────────────────────────
MAX_STEPS = 40
MAX_RETRIES_PER_TOOL = 3
AGENT_NUM_CTX = 32768
SUMMARY_NUM_CTX = 32768
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


def get_active_sessions() -> List[Dict]:
    with _sessions_lock:
        return [
            {"session_id": sid, "filename": ctx.filename, "step": ctx.current_step}
            for sid, ctx in _active_sessions.items()
        ]

# ── Tool schemas ──────────────────────────────────────────────────────────────
TOOL_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "probe_audio_fragment",
            "description": (
                "Transcribe a short audio fragment (30-90s) WITHOUT committing to a full run. "
                "Use this BEFORE transcribe_audio to: detect language, test VAD threshold, "
                "check audio quality, estimate optimal parameters. "
                "Returns sample text, confidence score, detected language, and parameter suggestions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number", "description": "Start offset in seconds (default 30 — skip intro silence)."},
                    "duration_sec": {"type": "number", "description": "Probe duration in seconds (30-90, default 60)."},
                    "language": {"type": "string", "description": "Language hint or 'auto'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_speaker_count",
            "description": (
                "Quickly estimate the number of speakers using pyannote on a 60-120s fragment. "
                "Call before diarize_audio to provide an accurate num_speakers hint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number", "description": "Start offset for analysis (default 0)."},
                    "duration_sec": {"type": "number", "description": "Fragment duration in seconds (60-120, default 90)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_transcription_params",
            "description": (
                "Store custom transcription parameters to be used by the next transcribe_audio call. "
                "Use after probe_audio_fragment to tune for the specific audio. "
                "Params persist in ctx until overridden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vad_filter_threshold": {"type": "number", "description": "VAD sensitivity 0.05-0.9 (lower=keep more speech, default 0.25). Lower for quiet/phone audio."},
                    "beam_size": {"type": "integer", "description": "Beam search width 1-10 (higher=better quality, slower, default 5)."},
                    "temperature": {"type": "number", "description": "Sampling temperature 0.0-1.0 (0=deterministic, >0=creative, default 0.0)."},
                    "no_speech_threshold": {"type": "number", "description": "Probability threshold 0.1-0.9 for discarding segments as silence (default 0.6)."},
                    "compression_ratio_threshold": {"type": "number", "description": "Repetition filter 1.5-3.5 (lower=stricter, default 2.4)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_audio_info",
            "description": (
                "Decode audio to PCM and return duration, filename, language hint. "
                "Call FIRST before any other tool."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": (
                "Run full ASR on the audio. Returns segments with text, timestamps, detected language. "
                "whisper=fastest/accurate/6GB; vibevoice=built-in diarization/9B/18GB; nemotron=multilingual/600M. "
                "Uses parameters set by set_transcription_params if called beforehand."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "enum": ["whisper", "vibevoice", "nemotron"],
                        "description": "ASR engine. whisper for most cases. vibevoice only when built-in diarization needed (18GB VRAM).",
                    },
                    "language": {"type": "string", "description": "Language hint or 'auto'."},
                },
                "required": ["engine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_transcript_quality",
            "description": (
                "Analyse transcript quality: compute per-segment confidence from word probabilities, "
                "detect hallucinations/repetitions, find long silence gaps, flag low-quality segments. "
                "Call after transcribe_audio to decide if re-transcription with different params is needed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_short_segments",
            "description": (
                "Merge consecutive same-speaker segments separated by less than gap_sec silence. "
                "Improves readability and diarization stability. Run after diarize_audio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gap_sec": {"type": "number", "description": "Max gap between segments to merge (default 0.8s)."},
                    "min_duration_sec": {"type": "number", "description": "Merge segments shorter than this (default 0.5s)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diarize_audio",
            "description": (
                "Assign speaker labels to transcript segments. "
                "Skip if vibevoice was used (built-in diarization). "
                "Use detect_speaker_count first for better accuracy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "num_speakers": {"type": "integer", "description": "Expected speaker count 1-10. 0=auto. Use detect_speaker_count result."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_speakers",
            "description": "Extract per-speaker: gender, age-group, emotion, speech rate, SNR.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_to_polish",
            "description": "Translate non-Polish segments to Polish via Ollama. Only if detected_language != 'pl' AND translation desired.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_speakers",
            "description": "Use LLM to infer real names from transcript. Sets display_name on profiles.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_partial_result",
            "description": (
                "Stream current transcript segments and speaker profiles to the UI immediately. "
                "Call this after diarize_audio or translate_to_polish to show users a live preview "
                "while the agent continues processing (entities, summary, RAG)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Optional status message for the user."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract named entities (persons, orgs, locations, dates, keywords) via Ollama. Handles long transcripts with windowed chunking.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_rag_index",
            "description": "Generate embeddings for transcript chunks via Ollama (nomic-embed-text). Required for semantic search and Q&A.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_transcript",
            "description": "Generate structured Polish-language summary. Uses map-reduce for recordings > 30 min.",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["brief", "detailed", "structured"],
                        "description": "structured=interview/meeting; brief=short clips; detailed=full narrative.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Execute Python code for custom analysis of the transcript. "
                "Variables available: segments (list of dicts), speaker_profiles (dict), "
                "display_names (dict), entities (dict or None), duration (float), detected_language (str). "
                "Set result=<value> to return a value. Output via print(). "
                "Use for: statistics, filtering, custom aggregations, quality checks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Must be safe analysis-only code."},
                    "description": {"type": "string", "description": "What this analysis computes."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_checkpoint",
            "description": (
                "Save current processing state (segments, profiles, entities, summary) to disk. "
                "Use before long Ollama operations to offload context and free agent working memory. "
                "Returns a checkpoint_id for loading later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Human-readable checkpoint label."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_checkpoint",
            "description": "Restore processing state from a previously saved checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_id": {"type": "string", "description": "Checkpoint ID returned by save_checkpoint."},
                },
                "required": ["checkpoint_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Persist a useful observation about this session to improve future decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string", "description": "What you observed (concrete, specific)."},
                    "improvement": {"type": "string", "description": "What to do differently next time."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Categorization tags."},
                },
                "required": ["observation", "improvement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal processing is complete. Call ONLY after transcription and all desired steps are done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Completion message for the user."},
                },
                "required": [],
            },
        },
    },
    # ── New tools ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_audio_quality",
            "description": (
                "Analyse audio signal quality: RMS level, dynamic range, clipping, silence ratio, noise floor. "
                "Call BEFORE setting transcription params to get data-driven VAD recommendations."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_translation_quality",
            "description": (
                "LLM-based quality assessment of a random sample of translations. "
                "Scores 1-5 per segment; returns worst segment IDs for targeted retry. "
                "Call after translate_to_polish."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_size": {"type": "integer", "description": "Number of segments to sample (default 6, max 10)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retranslate_segments",
            "description": (
                "Re-translate specific segments (by id) with a context-aware improved prompt. "
                "Use after validate_translation_quality identifies weak translations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_ids": {"type": "array", "items": {"type": "integer"}, "description": "Segment IDs to re-translate."},
                    "temperature": {"type": "number", "description": "Sampling temperature for retranslation (default 0.3 — slightly creative)."},
                },
                "required": ["segment_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_text_statistics",
            "description": (
                "Compute rich statistics: speaker balance, word counts, vocabulary richness, "
                "segment duration distribution, long segments. Use to decide if retranscription or "
                "re-diarization is needed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords_statistical",
            "description": (
                "Fast TF-IDF keyword extraction with no external dependencies. "
                "Returns top global keywords and per-speaker keywords. "
                "Much faster than LLM-based NER — use for quick topic overview."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_transcript",
            "description": "Fast text search with exact, case-insensitive, or fuzzy matching. Returns matching segments with scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "mode": {"type": "string", "enum": ["exact", "icase", "fuzzy"], "description": "Match mode (default: fuzzy)."},
                    "max_results": {"type": "integer", "description": "Max results to return (default 20)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_language_switches",
            "description": (
                "Use Whisper's language detector on evenly-spaced audio fragments to find code-switching. "
                "Returns language distribution and switch points."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_segments": {"type": "integer", "description": "Max segments to check (default 15)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_long_segments",
            "description": (
                "Split segments longer than max_sec seconds at sentence boundaries. "
                "Reduces LLM context load and improves diarization granularity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_sec": {"type": "number", "description": "Maximum segment duration in seconds (default 25)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_topics",
            "description": (
                "Time-windowed topic detection via LLM. Returns a topic label per N-minute window. "
                "Useful for chapter markers or navigation in long recordings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window_minutes": {"type": "number", "description": "Window size in minutes (default 5)."},
                    "max_windows": {"type": "integer", "description": "Max windows to process (default 8)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_speaker_labels",
            "description": (
                "Consolidate inconsistent speaker labels (format differences, rare speakers). "
                "Applies display_names if already identified. "
                "Merges speakers with < min_share of segments into dominant speaker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_share": {"type": "number", "description": "Merge speakers with < this fraction of segments (default 0.02)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_transcription_params",
            "description": (
                "Transcribe the same audio fragment with 2-3 different parameter sets and compare quality. "
                "Returns the best parameter set by avg word confidence. "
                "Use BEFORE transcribe_audio to find optimal settings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number", "description": "Start of test fragment (default 30s)."},
                    "duration_sec": {"type": "number", "description": "Duration to test (30-90s, default 60s)."},
                    "language": {"type": "string", "description": "Language hint or 'auto'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retranscribe_time_range",
            "description": (
                "Re-transcribe a specific time range (start_sec–end_sec) and replace existing segments. "
                "Use when verify_transcript_quality identifies a low-quality section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number", "description": "Start of range to re-transcribe."},
                    "end_sec": {"type": "number", "description": "End of range to re-transcribe."},
                    "language": {"type": "string", "description": "Language hint."},
                    "params": {"type": "object", "description": "Optional transcription params overrides."},
                },
                "required": ["start_sec", "end_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_artifact",
            "description": "Save a text artifact to the session storage. Useful for notes, intermediate analysis, plans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Artifact name (used as filename)."},
                    "content": {"type": "string", "description": "Text content to save."},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": "Read a previously saved artifact by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Artifact name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_artifacts",
            "description": "List all artifacts saved in the current session.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_speaker_assignments",
            "description": (
                "Post-diarization speaker refinement using three strategies: "
                "(1) neighbour majority vote for short segments, "
                "(2) micro-gap bridging for same-speaker runs, "
                "(3) LLM disambiguation for remaining ambiguous segments. "
                "Call AFTER diarize_audio and merge_short_segments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gap_sec": {"type": "number", "description": "Max gap for bridging (default 0.5s)."},
                    "short_sec": {"type": "number", "description": "Short segment threshold (default 1.5s)."},
                    "window": {"type": "integer", "description": "Neighbour window size (default 4)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_quotes_and_facts",
            "description": (
                "Extract verbatim quotes, facts, decisions, and key questions from the transcript. "
                "Quotes are injected as high-priority RAG entries. "
                "MANDATORY step — call before summarize_transcript."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_names_and_locations",
            "description": (
                "LLM-powered verification of extracted names and locations. "
                "Provides alternative spellings for uncertain entities, updates ctx.entities. "
                "Call after extract_entities — especially important for non-Polish audio."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_pass_transcribe_segment",
            "description": (
                "Re-transcribe a specific segment with context-primed initial_prompt and higher beam_size. "
                "Use on segments flagged by verify_transcript_quality as low-confidence. "
                "Only replaces if new transcription has higher confidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer", "description": "Segment index to re-transcribe."},
                    "padding_sec": {"type": "number", "description": "Context padding around segment (default 2.0s)."},
                },
                "required": ["segment_id"],
            },
        },
    },
]


# ── Agent shared context ──────────────────────────────────────────────────────

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

    # Current processing step (for status reporting)
    current_step: int = 0

    # Topics detected
    topics: List[Dict] = field(default_factory=list)

    # Per-segment quality scores (from verify_transcript_quality)
    segment_quality: Dict[int, float] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _send(ctx: AgentContext, event: dict) -> None:
    asyncio.run_coroutine_threadsafe(ctx.queue.put(event), ctx.loop)


def _progress(ctx: AgentContext, stage: str, pct: int, msg: str) -> None:
    _send(ctx, {"type": "progress", "stage": stage, "progress": pct, "message": msg})


def _agent_event(ctx: AgentContext, event_type: str, **kwargs) -> None:
    _send(ctx, {"type": event_type, **kwargs})


def _offload_ollama_model(model: str) -> None:
    """Send keep_alive=0 to release the model from Ollama GPU memory."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model,
                "messages": [],
                "keep_alive": 0,
            })
        logger.info("Offloaded Ollama model: %s", model)
    except Exception as exc:
        logger.debug("Offload request failed (non-fatal): %s", exc)


def _call_ollama_chat(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    num_ctx: int = AGENT_NUM_CTX,
    timeout: float = 180.0,
    model: Optional[str] = None,
) -> Optional[Dict]:
    body: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": num_ctx,
            "temperature": AGENT_TEMPERATURE,
            "top_p": 0.9,
        },
    }
    if tools:
        body["tools"] = tools
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{OLLAMA_URL}/api/chat", json=body)
            resp.raise_for_status()
            return resp.json().get("message")
    except Exception as exc:
        logger.warning("Ollama chat failed: %s", exc)
        return None


def _call_ollama_generate(
    prompt: str,
    num_ctx: int = SUMMARY_NUM_CTX,
    timeout: float = 360.0,
    json_format: Optional[Dict] = None,
    model: Optional[str] = None,
) -> str:
    body: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": num_ctx, "temperature": AGENT_TEMPERATURE},
    }
    if json_format:
        body["format"] = json_format
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{OLLAMA_URL}/api/generate", json=body)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("Ollama generate failed: %s", exc)
        return ""


def _call_ollama_embed(texts: List[str], timeout: float = 120.0, embed_model: Optional[str] = None) -> List[List[float]]:
    if not texts:
        return []
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": embed_model or OLLAMA_EMBEDDING_MODEL, "input": texts},
            )
            resp.raise_for_status()
            return resp.json().get("embeddings", [])
    except Exception as exc:
        logger.warning("Ollama embed failed: %s", exc)
        return []


def _decode_audio_bytes(content: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode audio bytes → (pcm_float32, sample_rate=16000)."""
    from pydub import AudioSegment
    import tempfile

    TARGET_SR = 16_000
    try:
        seg = AudioSegment.from_file(io.BytesIO(content))
    except Exception:
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            seg = AudioSegment.from_file(tmp)
        finally:
            os.unlink(tmp)
    seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2 ** (seg.sample_width * 8 - 1))
    return samples, TARGET_SR


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_probe_audio_fragment(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Transcribe a short fragment to calibrate parameters."""
    if ctx.audio_pcm is None:
        # Auto-decode first
        _tool_get_audio_info({}, ctx, models)

    start_sec = float(args.get("start_sec") or 30.0)
    duration_sec = min(float(args.get("duration_sec") or 60.0), 90.0)
    language = args.get("language") or ctx.language_hint

    sr = ctx.sample_rate
    start_smp = int(min(start_sec, max(0, ctx.duration - duration_sec)) * sr)
    end_smp = min(start_smp + int(duration_sec * sr), len(ctx.audio_pcm))
    fragment = ctx.audio_pcm[start_smp:end_smp]

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available for probe"}

    models.get("ensure_whisper_gpu", lambda: None)()

    _progress(ctx, "transcribing", 5, f"Próbkowanie {start_sec:.0f}–{start_sec+duration_sec:.0f}s…")

    try:
        probe_chunks, detected_lang, _ = whisper.transcribe(
            fragment, language, progress_cb=lambda *a: None,
            extra_kw={"beam_size": 3, "best_of": 1},  # fast probe
        )
    except Exception as exc:
        return {"error": str(exc)}

    # Analyse quality
    all_word_probs: List[float] = []
    texts: List[str] = []
    for ch in probe_chunks:
        texts.append(ch.get("text", ""))
        for w in (ch.get("words") or []):
            all_word_probs.append(float(w.get("probability", 1.0)))

    avg_conf = round(sum(all_word_probs) / len(all_word_probs), 3) if all_word_probs else 0.0
    sample_text = " ".join(texts)[:400]

    # Suggest params
    suggestions: Dict[str, Any] = {}
    if avg_conf < 0.6:
        suggestions["beam_size"] = 7
        suggestions["vad_filter_threshold"] = 0.15
        suggestions["note"] = "Low confidence — increase beam_size, lower VAD threshold"
    elif avg_conf < 0.75:
        suggestions["beam_size"] = 5
        suggestions["note"] = "Medium confidence — default params should work"
    else:
        suggestions["note"] = "Good confidence — default params are fine"

    if len(probe_chunks) == 0:
        suggestions["vad_filter_threshold"] = 0.1
        suggestions["note"] = "No segments detected — significantly lower VAD threshold"

    return {
        "detected_language": detected_lang,
        "segment_count": len(probe_chunks),
        "avg_confidence": avg_conf,
        "sample_text": sample_text,
        "fragment": f"{start_sec:.0f}–{start_sec+duration_sec:.0f}s",
        "suggestions": suggestions,
    }


def _tool_detect_speaker_count(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Estimate speaker count from a fragment using pyannote."""
    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    diarizer = models.get("diarizer")
    if diarizer is None or diarizer._method == "none":
        return {"error": "Diarizer (pyannote) not available", "estimated_speakers": 2}

    start_sec = float(args.get("start_sec") or 0.0)
    duration_sec = min(float(args.get("duration_sec") or 90.0), 120.0)
    sr = ctx.sample_rate
    start_smp = int(start_sec * sr)
    end_smp = min(start_smp + int(duration_sec * sr), len(ctx.audio_pcm))
    fragment = ctx.audio_pcm[start_smp:end_smp]

    _progress(ctx, "diarizing", 5, f"Wykrywanie liczby mówców ({start_sec:.0f}–{start_sec+duration_sec:.0f}s)…")

    # Run pyannote on fragment with dummy chunks
    dummy_chunks = [{"text": "", "start": 0.0, "end": len(fragment) / sr}]
    try:
        result = diarizer.diarize(fragment, sr, dummy_chunks)
        speakers = {s.get("speaker", "?") for s in result}
        n = len(speakers)
    except Exception as exc:
        logger.warning("Speaker count detection failed: %s", exc)
        return {"estimated_speakers": 2, "error": str(exc)}

    return {
        "estimated_speakers": n,
        "fragment": f"{start_sec:.0f}–{start_sec+duration_sec:.0f}s",
        "recommendation": f"Use num_speakers={n} in diarize_audio",
    }


def _tool_set_transcription_params(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Store tuning parameters to be used by the next transcribe_audio call."""
    allowed = {
        "vad_filter_threshold", "beam_size", "temperature",
        "no_speech_threshold", "compression_ratio_threshold",
    }
    stored = {}
    for k, v in args.items():
        if k in allowed and v is not None:
            ctx.transcription_params[k] = v
            stored[k] = v
    return {"stored_params": ctx.transcription_params, "applied": stored}


def _tool_get_audio_info(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Decode audio to PCM and return metadata."""
    import hashlib
    samples, sr = _decode_audio_bytes(ctx.audio_content, ctx.filename)
    ctx.audio_pcm = samples
    ctx.sample_rate = sr
    ctx.duration = len(samples) / sr
    ctx.audio_sha = hashlib.sha256(ctx.audio_content).hexdigest()
    return {
        "duration_seconds": round(ctx.duration, 1),
        "duration_minutes": round(ctx.duration / 60, 1),
        "filename": ctx.filename,
        "language_hint": ctx.language_hint or "auto",
        "translate_requested": ctx.do_translate,
        "sample_rate": sr,
    }


def _tool_transcribe_audio(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Run full ASR. Uses ctx.transcription_params if set. Streams segment_chunk events."""
    if ctx.audio_pcm is None:
        raise RuntimeError("Audio not decoded yet. Call get_audio_info first.")

    engine = args.get("engine", "whisper").lower()
    language = args.get("language") or ctx.language_hint
    ctx.asr_engine = engine

    def prog(pct, msg):
        _progress(ctx, "transcribing", pct, msg)

    # Streaming callback: emit raw segments chunk-by-chunk as Whisper processes audio
    _streaming_offset = [0]  # mutable counter for global segment index

    def on_chunk_segments(chunk_segs: List[Dict], offset_sec: float) -> None:
        if not chunk_segs:
            return
        start_idx = _streaming_offset[0]
        preview = [
            {
                "id": start_idx + i,
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": "—",  # pre-diarization placeholder
                "language": ctx.language_hint or "?",
            }
            for i, s in enumerate(chunk_segs)
        ]
        _streaming_offset[0] += len(chunk_segs)
        _send(ctx, {
            "type": "segment_chunk",
            "segments": preview,
            "offset_sec": offset_sec,
            "cumulative": _streaming_offset[0],
        })

    if engine == "vibevoice":
        transcriber = models["vibevoice_loader"](progress_cb=prog)
    elif engine == "nemotron":
        transcriber = models["nemotron_loader"](progress_cb=prog)
    else:
        models.get("ensure_whisper_gpu", lambda: None)()
        transcriber = models.get("whisper")

    _progress(ctx, "transcribing", 15, f"ASR: {engine} (params: {ctx.transcription_params or 'default'})…")

    extra_kw = dict(ctx.transcription_params) if ctx.transcription_params else None
    transcribe_kwargs: Dict[str, Any] = {"progress_cb": prog}
    if extra_kw and engine == "whisper":
        transcribe_kwargs["extra_kw"] = extra_kw
    if engine == "whisper":
        transcribe_kwargs["on_segments_cb"] = on_chunk_segments

    chunks, detected_lang, dur = transcriber.transcribe(ctx.audio_pcm, language, **transcribe_kwargs)
    if not chunks:
        raise ValueError("ASR returned no segments.")

    ctx.segments = chunks
    ctx.detected_language = detected_lang
    if dur:
        ctx.duration = dur

    return {
        "segment_count": len(chunks),
        "detected_language": detected_lang,
        "duration_seconds": round(ctx.duration, 1),
        "engine_used": engine,
        "params_used": ctx.transcription_params or "defaults",
    }


def _tool_verify_transcript_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Analyse transcript quality and emit a quality_report event."""
    if not ctx.segments:
        return {"error": "No segments to verify. Run transcribe_audio first."}

    low_conf: List[Dict] = []
    all_probs: List[float] = []
    repetitions = 0
    very_short = 0

    for i, seg in enumerate(ctx.segments):
        words = seg.get("words") or []
        probs = [float(w.get("probability", 1.0)) for w in words]
        if probs:
            seg_avg = sum(probs) / len(probs)
            all_probs.extend(probs)
            if seg_avg < 0.55:
                low_conf.append({"id": i, "start": seg.get("start"), "text": seg.get("text", "")[:60], "confidence": round(seg_avg, 3)})

        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur < 0.4:
            very_short += 1

        # Repetition check vs prev segment
        if i > 0:
            prev_text = ctx.segments[i - 1].get("text", "").lower().strip()
            curr_text = seg.get("text", "").lower().strip()
            if prev_text and curr_text and prev_text == curr_text:
                repetitions += 1

    avg_conf = round(sum(all_probs) / len(all_probs), 3) if all_probs else 1.0

    # Find long gaps
    gaps: List[Dict] = []
    for i in range(1, len(ctx.segments)):
        gap = (ctx.segments[i].get("start") or 0) - (ctx.segments[i - 1].get("end") or 0)
        if gap > 10.0:
            gaps.append({"before_segment": i, "gap_sec": round(gap, 1)})

    warnings: List[str] = []
    if avg_conf < 0.65:
        warnings.append(f"Low average confidence ({avg_conf:.2f}) — consider re-transcribing with higher beam_size or lower VAD threshold")
    if len(low_conf) > len(ctx.segments) * 0.2:
        warnings.append(f"{len(low_conf)} segments ({len(low_conf)*100//len(ctx.segments)}%) below confidence threshold")
    if repetitions > 3:
        warnings.append(f"{repetitions} consecutive duplicate segments detected — possible hallucinations")
    if len(gaps) > 0:
        warnings.append(f"{len(gaps)} long silence gaps (>10s) — verify audio continuity")

    ctx.quality_stats = {
        "avg_confidence": avg_conf,
        "low_confidence_count": len(low_conf),
        "repetitions": repetitions,
        "very_short_segments": very_short,
        "long_gaps": len(gaps),
        "total_segments": len(ctx.segments),
    }

    # Store per-segment confidence for UI visualization
    for i, seg in enumerate(ctx.segments):
        words = seg.get("words") or []
        probs = [float(w.get("probability", 1.0)) for w in words]
        if probs:
            ctx.segment_quality[i] = round(sum(probs) / len(probs), 3)

    _agent_event(ctx, "quality_report",
                 avg_confidence=avg_conf,
                 low_confidence_segments=low_conf[:10],
                 warnings=warnings,
                 gaps=gaps[:5],
                 stats=ctx.quality_stats)

    return {
        **ctx.quality_stats,
        "warnings": warnings,
        "recommend_retranscribe": avg_conf < 0.6 or repetitions > 5,
    }


def _tool_merge_short_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Merge very short or closely-spaced same-speaker segments."""
    if not ctx.segments:
        return {"skipped": True, "reason": "No segments"}

    gap_sec = float(args.get("gap_sec") or 0.8)
    min_dur = float(args.get("min_duration_sec") or 0.5)
    before = len(ctx.segments)

    merged: List[Dict] = []
    for seg in ctx.segments:
        if not merged:
            merged.append(dict(seg))
            continue

        prev = merged[-1]
        gap = (seg.get("start") or 0) - (prev.get("end") or 0)
        same_speaker = prev.get("speaker") == seg.get("speaker")
        prev_dur = (prev.get("end") or 0) - (prev.get("start") or 0)

        if same_speaker and (gap <= gap_sec or prev_dur < min_dur):
            # Merge
            prev["text"] = (prev.get("text", "") + " " + seg.get("text", "")).strip()
            prev["text_pl"] = (prev.get("text_pl", "") + " " + seg.get("text_pl", "")).strip() if prev.get("text_pl") else None
            prev["end"] = seg.get("end")
            words = (prev.get("words") or []) + (seg.get("words") or [])
            prev["words"] = words
        else:
            merged.append(dict(seg))

    # Re-index
    for i, s in enumerate(merged):
        s["id"] = i

    ctx.segments = merged
    return {"before": before, "after": len(merged), "merged_count": before - len(merged)}


def _tool_diarize_audio(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    if ctx.asr_engine == "vibevoice":
        return {"skipped": True, "reason": "vibevoice has built-in diarization"}

    diarizer = models.get("diarizer")
    if diarizer is None:
        return {"skipped": True, "reason": "diarizer not loaded"}

    num_speakers = int(args.get("num_speakers") or 0)
    _progress(ctx, "diarizing", 65, "Identyfikacja mówców (pyannote)…")
    ctx.segments = diarizer.diarize(
        ctx.audio_pcm, ctx.sample_rate, ctx.segments,
        num_speakers=num_speakers if num_speakers > 0 else None,
    )
    unique = len({s["speaker"] for s in ctx.segments})
    return {"speaker_count": unique}


def _tool_profile_speakers(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    profiler = models.get("profiler")
    audio_features = models.get("audio_features")
    _progress(ctx, "profiling", 74, "Analiza cech mówców…")
    if profiler:
        ctx.speaker_profiles_raw = profiler.profile_speakers(ctx.audio_pcm, ctx.sample_rate, ctx.segments)
    if audio_features and audio_features.loaded_extractors:
        try:
            ctx.audio_features_raw = audio_features.extract_per_speaker(ctx.audio_pcm, ctx.sample_rate, ctx.segments)
        except Exception as exc:
            logger.warning("Audio feature extraction failed: %s", exc)
    return {"speakers_profiled": list(ctx.speaker_profiles_raw.keys())}


def _tool_translate_to_polish(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Translation sub-agent: batches → stream → LLM quality check → targeted retry.

    The tool operates as a mini-loop:
    1. Translate all batches (streaming translation_chunk events to UI)
    2. Sample-check quality with LLM on every N-th batch
    3. Collect poor segments (score < 3) for a second pass
    4. Re-translate poor segments with an improved prompt
    5. Return quality report
    """
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    if ctx.detected_language == "pl" or not ctx.do_translate:
        for s in ctx.segments:
            s["text_pl"] = s["text"]
        return {"skipped": True, "reason": f"language={ctx.detected_language}, translate={ctx.do_translate}"}

    from .translator import translate_segments_to_polish

    quality_scores: List[float] = []
    low_quality_ids: List[int] = []
    batches_since_check = [0]

    def on_batch_done(batch_updates: List[Dict]) -> None:
        if not batch_updates:
            return
        _send(ctx, {"type": "translation_chunk", "updates": batch_updates})
        batches_since_check[0] += 1

        # Quality-check every 3rd batch (≈45 segments)
        if batches_since_check[0] % 3 != 0 or ctx.detected_language == "auto":
            return

        # Sample 3 segments from this batch
        sample = batch_updates[:3]
        if not sample:
            return

        src_lang = ctx.detected_language
        lines = "\n".join(
            f"{i+1}. [{src_lang}] {ctx.segments[u['idx']].get('text','')[:120]}\n"
            f"   [pl]   {u['text_pl'][:120]}"
            for i, u in enumerate(sample)
        )
        qprompt = (
            f"Oceń jakość tłumaczeń z {src_lang} na polski. "
            "Dla każdej pary podaj ocenę 1-5 (1=błąd/inny język, 3=OK, 5=świetne naturalne tłumaczenie).\n\n"
            f"{lines}\n\n"
            "Zwróć JSON: {\"scores\": [int, ...], \"avg\": float, \"issues\": [str, ...]}"
        )
        qschema = {
            "type": "object",
            "properties": {
                "scores": {"type": "array", "items": {"type": "integer"}},
                "avg": {"type": "number"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scores", "avg"],
        }
        try:
            raw = _call_ollama_generate(qprompt, json_format=qschema, num_ctx=8192, timeout=60.0, model=ctx.ollama_model)
            if raw:
                data = json.loads(raw)
                avg = float(data.get("avg", 3.0))
                quality_scores.append(avg)
                scores = data.get("scores", [])
                for i, sc in enumerate(scores):
                    if sc <= 2 and i < len(sample):
                        low_quality_ids.append(sample[i]["idx"])
                _send(ctx, {"type": "translation_quality_check",
                            "batch_avg": round(avg, 2), "issues": data.get("issues", [])})
        except Exception as exc:
            logger.debug("Translation quality check failed (non-fatal): %s", exc)

    _progress(ctx, "translating", 82, f"Tłumaczenie sub-agenta ({ctx.detected_language}→pl)…")
    ctx.segments = translate_segments_to_polish(
        ctx.segments, ctx.detected_language, on_batch_done=on_batch_done, model=ctx.ollama_model
    )

    # ── Second pass: re-translate poor segments with improved prompt ──────────
    retranslated = 0
    if low_quality_ids:
        _progress(ctx, "translating", 94, f"Ponowne tłumaczenie {len(low_quality_ids)} słabych segmentów…")
        for idx in set(low_quality_ids):
            if idx >= len(ctx.segments):
                continue
            seg = ctx.segments[idx]
            orig = seg.get("text", "")
            retry_prompt = (
                f"Przetłumacz poniższe zdanie z języka {ctx.detected_language} na naturalny, poprawny POLSKI. "
                "Oddaj sens i styl. Odpowiedz TYLKO polskim tłumaczeniem, bez komentarzy.\n\n"
                f"Zdanie: {orig}"
            )
            try:
                new_pl = _call_ollama_generate(retry_prompt, num_ctx=4096, timeout=30.0, model=ctx.ollama_model)
                if new_pl:
                    ctx.segments[idx]["text_pl"] = new_pl.strip()
                    retranslated += 1
                    _send(ctx, {"type": "translation_chunk",
                                "updates": [{"idx": idx, "text_pl": ctx.segments[idx]["text_pl"]}]})
            except Exception as exc:
                logger.debug("Retranslation of segment %d failed: %s", idx, exc)

    overall_avg = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None
    translated = sum(1 for s in ctx.segments if s.get("text_pl"))
    return {
        "segments_translated": translated,
        "source_language": ctx.detected_language,
        "quality_checks_performed": len(quality_scores),
        "overall_avg_quality": overall_avg,
        "low_quality_retranslated": retranslated,
        "quality_acceptable": (overall_avg is None or overall_avg >= 3.0),
    }


# ── New tools ─────────────────────────────────────────────────────────────────

def _tool_analyze_audio_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Analyse audio signal: RMS, dynamic range, clipping, silence ratio, noise floor."""
    if ctx.audio_pcm is None:
        return {"error": "Audio not decoded. Run get_audio_info first."}

    audio = ctx.audio_pcm
    sr = ctx.sample_rate

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    dynamic_range_db = round(20 * np.log10(peak / (rms + 1e-10)), 1)
    clipping_ratio = float(np.mean(np.abs(audio) > 0.98))

    # Frame-based energy for silence / noise estimation
    frame_size = int(0.025 * sr)  # 25 ms
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, frame_size)]
    if frames:
        energies = np.array([np.mean(f ** 2) for f in frames])
        noise_floor_db = round(10 * np.log10(float(np.percentile(energies, 10)) + 1e-10), 1)
        silence_threshold = float(np.percentile(energies, 20))
        silence_ratio = round(float(np.mean(energies < silence_threshold)), 3)
    else:
        noise_floor_db = -60.0
        silence_ratio = 0.0

    warnings: List[str] = []
    recommendations: List[str] = []
    if clipping_ratio > 0.001:
        warnings.append(f"Audio clipping: {clipping_ratio*100:.2f}% of samples saturated")
    if rms < 0.01:
        warnings.append("Very low volume level")
        recommendations.append("Use vad_filter_threshold=0.10 or 0.15 — lower sensitivity for quiet audio")
    if silence_ratio > 0.65:
        warnings.append(f"High silence ratio ({silence_ratio:.0%}) — long pauses or sparse speech")
        recommendations.append("Increase min_silence_duration_ms=600 and min_speech_duration_ms=200")
    if dynamic_range_db < 6:
        warnings.append("Low dynamic range — compressed or phone-call audio")
        recommendations.append("Consider beam_size=7 for compressed audio")

    return {
        "rms_level": round(rms, 5),
        "peak_level": round(peak, 4),
        "dynamic_range_db": dynamic_range_db,
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": silence_ratio,
        "noise_floor_db": noise_floor_db,
        "duration_sec": round(ctx.duration, 1),
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _tool_validate_translation_quality(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    LLM-based sample quality assessment of the current translations.
    Scores 1-5 per segment sample; returns worst segment IDs for targeted retry.
    """
    translated = [s for s in ctx.segments if s.get("text_pl") and s.get("text_pl") != s.get("text")]
    if not translated:
        return {"skipped": True, "reason": "No translations yet. Run translate_to_polish first."}

    import random
    n_sample = min(int(args.get("sample_size", 6)), 10, len(translated))
    sample = random.sample(translated, n_sample)

    src_lang = ctx.detected_language
    lines = "\n".join(
        f"{i+1}. [{src_lang}] {s.get('text','')[:150]}\n   [pl]   {s.get('text_pl','')[:150]}"
        for i, s in enumerate(sample)
    )
    prompt = (
        f"Jesteś ekspertem od tłumaczeń. Oceń poniższe tłumaczenia z {src_lang} na polski.\n"
        f"Skala: 1=kompletnie błędne/inny język, 2=bardzo złe, 3=akceptowalne, 4=dobre, 5=świetne.\n\n"
        f"{lines}\n\n"
        "Zwróć JSON: {\"scores\": [int,...], \"avg_score\": float, "
        "\"worst_indices\": [int,...], \"issues\": [str,...]}"
    )
    json_schema = {
        "type": "object",
        "properties": {
            "scores": {"type": "array", "items": {"type": "integer"}},
            "avg_score": {"type": "number"},
            "worst_indices": {"type": "array", "items": {"type": "integer"}},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scores", "avg_score"],
    }
    raw = _call_ollama_generate(prompt, json_format=json_schema, num_ctx=SUMMARY_NUM_CTX, timeout=90.0, model=ctx.ollama_model)
    if not raw:
        return {"error": "LLM did not respond"}

    try:
        data = json.loads(raw)
        avg = float(data.get("avg_score", 0.0))
        worst_local = [i for i in data.get("worst_indices", []) if i < len(sample)]
        worst_ids = [sample[i].get("id", sample[i].get("idx", 0)) for i in worst_local]
        return {
            "sample_size": n_sample,
            "avg_score": round(avg, 2),
            "scores": data.get("scores", []),
            "issues": data.get("issues", []),
            "worst_segment_ids": worst_ids,
            "needs_retranslation": avg < 3.0,
            "recommendation": (
                f"Call retranslate_segments with segment_ids={worst_ids}"
                if avg < 3.5 and worst_ids else "Translation quality acceptable"
            ),
        }
    except (json.JSONDecodeError, KeyError) as exc:
        return {"error": f"Parse error: {exc}", "raw": raw[:200]}


def _tool_retranslate_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Re-translate specific segments (by id) with an improved prompt.
    Use after validate_translation_quality identifies weak segments.
    """
    segment_ids: List[int] = [int(x) for x in (args.get("segment_ids") or [])]
    temperature = float(args.get("temperature", 0.3))  # slightly creative for quality
    if not segment_ids:
        return {"error": "segment_ids required"}
    if not ctx.segments:
        return {"error": "No segments"}

    id_map = {s.get("id", i): i for i, s in enumerate(ctx.segments)}
    retranslated: List[int] = []
    src_lang = ctx.detected_language

    for seg_id in segment_ids:
        seg_idx = id_map.get(seg_id)
        if seg_idx is None:
            continue
        seg = ctx.segments[seg_idx]
        orig = seg.get("text", "")
        context_prev = ctx.segments[seg_idx - 1].get("text", "") if seg_idx > 0 else ""
        context_next = ctx.segments[seg_idx + 1].get("text", "") if seg_idx < len(ctx.segments) - 1 else ""

        prompt = (
            f"Przetłumacz poniższe zdanie z {src_lang} na NATURALNY POLSKI.\n"
            "Weź pod uwagę kontekst (zdania przed i po).\n"
            "Odpowiedz TYLKO gotowym polskim tłumaczeniem bez żadnych komentarzy.\n\n"
            + (f"Poprzednie zdanie (kontekst): {context_prev}\n" if context_prev else "")
            + f"ZDANIE DO PRZETŁUMACZENIA: {orig}\n"
            + (f"Następne zdanie (kontekst): {context_next}\n" if context_next else "")
        )
        try:
            body: Dict[str, Any] = {
                "model": ctx.ollama_model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": 4096, "temperature": temperature},
            }
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(f"{OLLAMA_URL}/api/generate", json=body)
                resp.raise_for_status()
                new_pl = resp.json().get("response", "").strip()
            if new_pl:
                ctx.segments[seg_idx]["text_pl"] = new_pl
                retranslated.append(seg_id)
                _send(ctx, {"type": "translation_chunk", "updates": [{"idx": seg_idx, "text_pl": new_pl}]})
        except Exception as exc:
            logger.warning("Retranslation segment %d failed: %s", seg_id, exc)

    return {"retranslated_count": len(retranslated), "segment_ids": retranslated}


def _tool_compute_text_statistics(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Compute rich statistics: speaker balance, vocabulary, word count, speaking rate,
    segment duration distribution. Useful for quality decisions.
    """
    if not ctx.segments:
        return {"error": "No segments"}

    import re

    total_words = 0
    speaker_stats: Dict[str, Dict] = {}

    for seg in ctx.segments:
        text = (seg.get("text_pl") or seg.get("text") or "")
        words = re.findall(r'\b\w+\b', text)
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        sp = seg.get("speaker", "?")

        if sp not in speaker_stats:
            speaker_stats[sp] = {"words": 0, "segments": 0, "duration_sec": 0.0, "chars": 0}
        speaker_stats[sp]["words"] += len(words)
        speaker_stats[sp]["segments"] += 1
        speaker_stats[sp]["duration_sec"] += dur
        speaker_stats[sp]["chars"] += len(text)
        total_words += len(words)

    # Vocabulary richness (type-token ratio on sample)
    all_text = " ".join((s.get("text_pl") or s.get("text") or "") for s in ctx.segments)
    all_words_lower = re.findall(r'\b\w+\b', all_text.lower())
    vocab_size = len(set(all_words_lower))
    ttr = round(vocab_size / max(len(all_words_lower), 1), 4)

    # Segment duration distribution
    durations = [(s.get("end") or 0) - (s.get("start") or 0) for s in ctx.segments]
    avg_dur = round(float(np.mean(durations)), 2) if durations else 0
    max_dur = round(float(np.max(durations)), 2) if durations else 0
    long_segs = [i for i, d in enumerate(durations) if d > 30]

    # Speaker balance
    dominant = max(speaker_stats, key=lambda s: speaker_stats[s]["duration_sec"], default=None)
    speaker_shares = {
        sp: round(v["duration_sec"] / max(ctx.duration, 1), 3)
        for sp, v in speaker_stats.items()
    }

    return {
        "total_words": total_words,
        "total_segments": len(ctx.segments),
        "vocabulary_size": vocab_size,
        "type_token_ratio": ttr,
        "avg_segment_duration_sec": avg_dur,
        "max_segment_duration_sec": max_dur,
        "long_segments_over_30s": len(long_segs),
        "long_segment_indices": long_segs[:5],
        "speaker_word_counts": {sp: v["words"] for sp, v in speaker_stats.items()},
        "speaker_duration_share": speaker_shares,
        "dominant_speaker": dominant,
        "speaker_balance_warning": (
            f"{dominant} dominates ({speaker_shares.get(dominant, 0):.0%} of audio)"
            if dominant and speaker_shares.get(dominant, 0) > 0.85
            else None
        ),
    }


def _tool_extract_keywords_statistical(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Fast TF-IDF keyword extraction — no external dependencies.
    Returns top keywords globally and per speaker.
    """
    if not ctx.segments:
        return {"error": "No segments"}

    import re
    import math
    from collections import Counter

    # Polish + English stopwords
    STOP = {
        'i', 'w', 'z', 'na', 'do', 'nie', 'się', 'to', 'że', 'a', 'jest', 'jak', 'tak',
        'ale', 'dla', 'po', 'co', 'by', 'już', 'go', 'jej', 'jego', 'ich', 'ten', 'ta',
        'te', 'tego', 'tej', 'tym', 'te', 'są', 'być', 'ma', 'mam', 'pan', 'pani',
        'the', 'and', 'to', 'of', 'in', 'is', 'it', 'that', 'was', 'for', 'on', 'are',
        'as', 'at', 'be', 'by', 'he', 'she', 'we', 'or', 'an', 'if', 'my', 'no', 'so',
        'więc', 'też', 'tam', 'tu', 'tu', 'czy', 'który', 'która', 'które', 'kiedy',
    }

    def tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r'\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}\b', text.lower())
                if t not in STOP]

    # Per-segment tokenized docs (for IDF)
    docs = [tokenize(s.get("text_pl") or s.get("text") or "") for s in ctx.segments]
    N = len(docs)
    doc_freq: Counter = Counter()
    for doc in docs:
        for w in set(doc):
            doc_freq[w] += 1

    # Corpus-wide TF-IDF
    corpus_tf: Counter = Counter()
    for doc in docs:
        corpus_tf.update(doc)

    tfidf_scores: Dict[str, float] = {
        w: (cnt / max(sum(corpus_tf.values()), 1)) * math.log((N + 1) / (doc_freq[w] + 1) + 1)
        for w, cnt in corpus_tf.items()
    }
    top_global = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)[:25]  # type: ignore[arg-type]

    # Per-speaker keywords
    speaker_kw: Dict[str, List[str]] = {}
    for sp in {s.get("speaker") for s in ctx.segments}:
        sp_docs = [tokenize(s.get("text_pl") or s.get("text") or "")
                   for s in ctx.segments if s.get("speaker") == sp]
        sp_tf: Counter = Counter(w for doc in sp_docs for w in doc)
        sp_scores = {
            w: (cnt / max(sum(sp_tf.values()), 1)) * math.log((N + 1) / (doc_freq[w] + 1) + 1)
            for w, cnt in sp_tf.items()
        }
        speaker_kw[sp or "?"] = sorted(sp_scores, key=sp_scores.get, reverse=True)[:10]  # type: ignore[arg-type]

    return {
        "top_keywords": top_global,
        "keyword_tfidf_scores": {k: round(v, 5) for k, v in tfidf_scores.items()
                                  if k in top_global[:15]},
        "speaker_keywords": speaker_kw,
        "total_unique_terms": len(tfidf_scores),
    }


def _tool_search_in_transcript(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Fast text search across segments. Supports exact, case-insensitive, and fuzzy modes.
    Returns matching segment IDs with context.
    """
    if not ctx.segments:
        return {"error": "No segments"}

    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query parameter required"}

    mode = args.get("mode", "fuzzy")  # exact | icase | fuzzy
    max_results = int(args.get("max_results", 20))

    import difflib
    import re

    results: List[Dict] = []
    q_lower = query.lower()

    for i, seg in enumerate(ctx.segments):
        text = seg.get("text_pl") or seg.get("text") or ""
        text_lower = text.lower()
        score = 0.0

        if mode == "exact":
            if query in text:
                score = 1.0
        elif mode == "icase":
            if q_lower in text_lower:
                score = 1.0
        else:  # fuzzy
            # difflib ratio on sliding windows
            words = text_lower.split()
            q_words = q_lower.split()
            n = len(q_words)
            for j in range(max(1, len(words) - n + 1)):
                window = " ".join(words[j:j + n + 2])
                r = difflib.SequenceMatcher(None, q_lower, window).ratio()
                if r > score:
                    score = r

        if score >= (0.6 if mode == "fuzzy" else 1.0):
            results.append({
                "segment_id": seg.get("id", i),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker"),
                "text": text[:200],
                "score": round(score, 3),
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {
        "query": query,
        "mode": mode,
        "match_count": len(results),
        "results": results[:max_results],
    }


def _tool_detect_language_switches(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Detect language switches across segments using Whisper's language detection.
    Samples evenly-spaced audio fragments, returns switches and language distribution.
    """
    if ctx.audio_pcm is None or not ctx.segments:
        return {"error": "Audio and segments required."}

    whisper = models.get("whisper")
    if whisper is None or whisper.model is None:
        return {"skipped": True, "reason": "Whisper not available"}

    models.get("ensure_whisper_gpu", lambda: None)()

    max_check = min(int(args.get("max_segments", 15)), len(ctx.segments), 15)
    step = max(1, len(ctx.segments) // max_check)
    indices = list(range(0, len(ctx.segments), step))[:max_check]

    sr = ctx.sample_rate
    languages_found: Dict[str, int] = {}
    switches: List[Dict] = []
    prev_lang = ctx.detected_language

    for idx in indices:
        seg = ctx.segments[idx]
        start_smp = int((seg.get("start") or 0) * sr)
        end_smp = int((seg.get("end") or 0) * sr)
        fragment = ctx.audio_pcm[start_smp:end_smp]
        if len(fragment) < sr // 2:
            continue
        try:
            _, probs = whisper.model.detect_language(fragment)
            lang = max(probs, key=probs.get)  # type: ignore[arg-type]
            conf = float(probs[lang])
            languages_found[lang] = languages_found.get(lang, 0) + 1
            if lang != prev_lang and conf > 0.65:
                switches.append({
                    "segment_id": seg.get("id", idx),
                    "start": seg.get("start"),
                    "from_lang": prev_lang,
                    "to_lang": lang,
                    "confidence": round(conf, 3),
                })
            prev_lang = lang
        except Exception as exc:
            logger.debug("Language detection failed for segment %d: %s", idx, exc)

    return {
        "segments_checked": len(indices),
        "languages_found": languages_found,
        "code_switches": switches,
        "is_multilingual": len(languages_found) > 1,
        "dominant_language": max(languages_found, key=languages_found.get) if languages_found else ctx.detected_language,  # type: ignore[arg-type]
    }


def _tool_split_long_segments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Split segments longer than max_sec seconds at sentence boundaries (period/comma).
    Reduces memory requirements for LLM processing and improves diarization.
    """
    if not ctx.segments:
        return {"skipped": True}

    max_sec = float(args.get("max_sec", 25.0))
    import re

    before = len(ctx.segments)
    new_segs: List[Dict] = []

    for seg in ctx.segments:
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur <= max_sec:
            new_segs.append(seg)
            continue

        text = seg.get("text", "")
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
        if len(sentences) < 2:
            new_segs.append(seg)
            continue

        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_dur = seg_end - seg_start
        total_chars = max(sum(len(s) for s in sentences), 1)
        offset = seg_start

        for sent in sentences:
            sent_dur = seg_dur * len(sent) / total_chars
            new_segs.append({
                **seg,
                "text": sent,
                "text_pl": None,
                "start": round(offset, 3),
                "end": round(offset + sent_dur, 3),
                "words": [],
            })
            offset += sent_dur

    for i, s in enumerate(new_segs):
        s["id"] = i

    ctx.segments = new_segs
    return {"before": before, "after": len(new_segs), "split_count": len(new_segs) - before}


def _tool_detect_topics(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Time-windowed topic detection via LLM. Identifies main topics per N-minute window.
    Useful for generating chapter markers or navigation.
    """
    if not ctx.segments:
        return {"error": "No segments. Run transcribe_audio first."}

    window_min = float(args.get("window_minutes", 5.0))
    window_sec = window_min * 60
    max_windows = int(args.get("max_windows", 8))

    # Build time windows
    windows: List[Dict] = []
    current_segs: List[str] = []
    current_start = 0.0
    prev_end = 0.0

    for seg in ctx.segments:
        t = seg.get("start") or 0
        if t - current_start >= window_sec and current_segs:
            windows.append({"start": current_start, "end": prev_end,
                            "text": " ".join(current_segs)[:1000]})
            current_segs = []
            current_start = t
            if len(windows) >= max_windows:
                break
        current_segs.append((seg.get("text_pl") or seg.get("text") or "").strip())
        prev_end = seg.get("end") or t

    if current_segs:
        windows.append({"start": current_start, "end": prev_end,
                        "text": " ".join(current_segs)[:1000]})

    _progress(ctx, "extracting", 75, f"Wykrywanie tematów ({len(windows)} okien)…")

    topics: List[Dict] = []
    for win in windows[:max_windows]:
        prompt = (
            "Podaj główny temat poniższego fragmentu rozmowy w 1 krótkim zdaniu (max 15 słów) po polsku.\n\n"
            f"Fragment:\n{win['text']}\n\nTemat:"
        )
        try:
            topic = _call_ollama_generate(prompt, num_ctx=4096, timeout=30.0, model=ctx.ollama_model)
            topics.append({
                "start_sec": round(win["start"], 1),
                "end_sec": round(win["end"], 1),
                "topic": topic.strip()[:120] if topic else "—",
            })
        except Exception:
            topics.append({"start_sec": round(win["start"], 1), "end_sec": round(win["end"], 1), "topic": "—"})

    return {"windows": len(topics), "topics": topics}


def _tool_normalize_speaker_labels(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Consolidate inconsistent speaker labels (e.g. GŁOS_01 and GŁOS_1 → GŁOS_01).
    Also renames speakers based on display_names if available.
    Optionally reassigns rare speakers (< min_share of segments) to nearest majority speaker.
    """
    if not ctx.segments:
        return {"skipped": True}

    from collections import Counter
    import re

    min_share = float(args.get("min_share", 0.02))  # merge speakers with < 2% of segments

    # Normalize format: GŁOS_1 → GŁOS_01, SPEAKER_00 → GŁOS_01 etc.
    def normalize(label: str) -> str:
        m = re.search(r'\d+', label)
        num = int(m.group()) if m else 0
        return f"GŁOS_{num:02d}"

    label_map: Dict[str, str] = {}
    for seg in ctx.segments:
        sp = seg.get("speaker", "")
        if sp and sp not in label_map:
            label_map[sp] = normalize(sp)

    sp_counts = Counter(s.get("speaker") for s in ctx.segments)
    total = len(ctx.segments)
    rare = {sp for sp, cnt in sp_counts.items() if cnt / total < min_share}

    changes = 0
    for seg in ctx.segments:
        orig = seg.get("speaker", "")
        new = label_map.get(orig, orig)

        # Apply display name if known
        if orig in ctx.display_names:
            new = ctx.display_names[orig]

        # Merge rare speaker into closest (dominant)
        if orig in rare and len(sp_counts) > 1:
            dominant = max((s for s in sp_counts if s not in rare), key=sp_counts.get, default=None)
            if dominant:
                new = label_map.get(dominant, dominant)

        if seg.get("speaker") != new:
            seg["speaker"] = new
            changes += 1

    return {"changes": changes, "label_map": label_map, "rare_merged": list(rare)}


def _tool_compare_transcription_params(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Transcribe the same 60s audio fragment with 2 different parameter sets and compare quality.
    Returns the best parameter set based on avg word confidence.
    Use the winner with set_transcription_params before running transcribe_audio.
    """
    if ctx.audio_pcm is None:
        _tool_get_audio_info({}, ctx, models)

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available"}
    models.get("ensure_whisper_gpu", lambda: None)()

    start_sec = float(args.get("start_sec", 30.0))
    duration_sec = min(float(args.get("duration_sec", 60.0)), 90.0)
    language = args.get("language") or ctx.language_hint

    sr = ctx.sample_rate
    start_smp = int(min(start_sec, max(0.0, ctx.duration - duration_sec)) * sr)
    end_smp = min(start_smp + int(duration_sec * sr), len(ctx.audio_pcm))
    fragment = ctx.audio_pcm[start_smp:end_smp]

    # Default param sets — expand based on audio quality signals
    param_sets = [
        {"label": "default", "beam_size": 5, "vad_filter_threshold": 0.25, "temperature": 0.0},
        {"label": "sensitive_vad", "beam_size": 5, "vad_filter_threshold": 0.10, "temperature": 0.0},
        {"label": "high_beam", "beam_size": 8, "vad_filter_threshold": 0.20, "temperature": 0.0},
    ]
    # Allow caller to pass custom sets
    custom = args.get("param_sets")
    if isinstance(custom, list) and len(custom) >= 2:
        param_sets = custom[:3]

    _progress(ctx, "transcribing", 5, f"Porównywanie {len(param_sets)} zestawów parametrów na {duration_sec:.0f}s próbce…")

    results = []
    for ps in param_sets:
        label = ps.pop("label", "set")
        try:
            extra = dict(ps)
            chunks, lang, _ = whisper.transcribe(
                fragment, language, progress_cb=lambda *a: None, extra_kw=extra
            )
            all_probs = [float(w.get("probability", 1.0))
                         for ch in chunks for w in (ch.get("words") or [])]
            avg_conf = round(sum(all_probs) / len(all_probs), 3) if all_probs else 0.0
            sample = " ".join(ch.get("text", "") for ch in chunks[:3])[:200]
            results.append({
                "label": label,
                "params": ps,
                "segment_count": len(chunks),
                "avg_confidence": avg_conf,
                "detected_language": lang,
                "sample_text": sample,
            })
        except Exception as exc:
            results.append({"label": label, "params": ps, "error": str(exc)})

    # Pick winner
    valid = [r for r in results if "avg_confidence" in r]
    winner = max(valid, key=lambda r: r["avg_confidence"]) if valid else None

    return {
        "fragment": f"{start_sec:.0f}–{start_sec+duration_sec:.0f}s",
        "results": results,
        "best_params": winner["params"] if winner else None,
        "best_label": winner["label"] if winner else None,
        "recommendation": (
            f"Call set_transcription_params with {winner['params']} "
            f"(avg_confidence={winner['avg_confidence']})"
            if winner else "All param sets failed"
        ),
    }


def _tool_retranscribe_time_range(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Re-transcribe a specific time range and replace/supplement existing segments.
    Useful when verify_transcript_quality reveals a problematic section.
    """
    if ctx.audio_pcm is None:
        return {"error": "Audio not decoded. Run get_audio_info first."}

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available for retranscription"}
    models.get("ensure_whisper_gpu", lambda: None)()

    start_sec = float(args.get("start_sec", 0.0))
    end_sec = float(args.get("end_sec", min(start_sec + 60, ctx.duration)))
    language = args.get("language") or ctx.language_hint
    extra_kw = args.get("params") or {}

    sr = ctx.sample_rate
    start_smp = int(start_sec * sr)
    end_smp = int(end_sec * sr)
    fragment = ctx.audio_pcm[start_smp:end_smp]

    _progress(ctx, "transcribing", 5, f"Re-transkrypcja {start_sec:.0f}–{end_sec:.0f}s…")

    try:
        new_chunks, detected_lang, _ = whisper.transcribe(
            fragment, language, progress_cb=lambda *a: None,
            extra_kw=extra_kw if extra_kw else None,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if not new_chunks:
        return {"replaced": 0, "note": "No segments returned from re-transcription"}

    # Adjust timestamps to absolute
    for ch in new_chunks:
        ch["start"] = round(ch["start"] + start_sec, 3)
        ch["end"] = round(ch["end"] + start_sec, 3)
        for w in (ch.get("words") or []):
            w["start"] = round(w["start"] + start_sec, 3)
            w["end"] = round(w["end"] + start_sec, 3)

    # Remove existing segments in this time range
    original_count = len(ctx.segments)
    ctx.segments = [s for s in ctx.segments
                    if not (start_sec <= (s.get("start") or 0) < end_sec)]

    # Insert new segments and sort
    ctx.segments.extend(new_chunks)
    ctx.segments.sort(key=lambda s: s.get("start") or 0)

    # Re-index
    for i, s in enumerate(ctx.segments):
        s["id"] = i

    return {
        "replaced_range": f"{start_sec:.1f}–{end_sec:.1f}s",
        "segments_removed": original_count - (len(ctx.segments) - len(new_chunks)),
        "new_segments": len(new_chunks),
        "total_segments": len(ctx.segments),
        "detected_language": detected_lang,
    }


def _tool_write_artifact(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Save a text artifact to the session directory for later use or reference."""
    name = (args.get("name") or "artifact").strip().replace("/", "_").replace("..", "")
    content = args.get("content", "")
    if not name:
        return {"error": "name required"}

    session_dir = ARTIFACTS_DIR / ctx.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.txt"
    try:
        path.write_text(str(content), encoding="utf-8")
        return {"saved": True, "path": str(path), "bytes": len(content)}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_read_artifact(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Read a previously saved artifact."""
    name = (args.get("name") or "").strip().replace("/", "_").replace("..", "")
    if not name:
        return {"error": "name required"}
    path = ARTIFACTS_DIR / ctx.session_id / f"{name}.txt"
    if not path.exists():
        return {"error": f"Artifact '{name}' not found"}
    try:
        content = path.read_text(encoding="utf-8")
        return {"name": name, "content": content[:5000], "truncated": len(content) > 5000}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_list_artifacts(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """List all artifacts saved in the current session."""
    session_dir = ARTIFACTS_DIR / ctx.session_id
    if not session_dir.exists():
        return {"artifacts": []}
    files = [f.stem for f in session_dir.glob("*.txt")]
    return {"artifacts": files, "count": len(files)}


def _tool_identify_speakers(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    from .speaker_identifier import identify_speakers
    _progress(ctx, "identifying", 94, "Identyfikacja mówców (LLM)…")
    try:
        ctx.display_names = identify_speakers(ctx.segments, ctx.speaker_profiles_raw)
    except Exception as exc:
        logger.warning("Speaker identification failed: %s", exc)
        ctx.display_names = {}
    return {"display_names": ctx.display_names}


def _tool_emit_partial_result(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Stream current segments to UI without finishing processing."""
    if not ctx.segments:
        return {"skipped": True, "reason": "No segments yet"}

    speaker_profiles = _build_speaker_profiles(ctx)
    segments_out = _build_segments_out(ctx)

    _send(ctx, {
        "type": "partial_segments",
        "segments": segments_out,
        "speaker_profiles": speaker_profiles,
        "detected_language": ctx.detected_language,
        "duration": round(ctx.duration, 2),
        "message": args.get("message", "Podgląd wstępny…"),
    })
    ctx.partial_emitted = True
    return {"emitted": len(segments_out), "message": args.get("message", "")}


def _tool_extract_entities(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")

    full_text = " ".join((s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments)
    WINDOW_CHARS, OVERLAP, MAX_WINDOWS = 5000, 500, 6
    windows: List[str] = []
    if len(full_text) <= WINDOW_CHARS:
        windows = [full_text]
    else:
        step = WINDOW_CHARS - OVERLAP
        for i in range(0, len(full_text), step):
            if len(windows) >= MAX_WINDOWS:
                break
            windows.append(full_text[i: i + WINDOW_CHARS])

    lang_names = {"en": "angielski", "ru": "rosyjski", "uk": "ukraiński", "de": "niemiecki", "fr": "francuski"}
    lang_note = (f"Tekst pochodzi z języka {lang_names.get(ctx.detected_language, ctx.detected_language)}. "
                 if ctx.detected_language not in ("pl", "auto") else "")

    _progress(ctx, "extracting", 80, f"Ekstrakcja encji ({len(windows)} okien)…")
    merged: Dict[str, set] = {k: set() for k in ("persons", "organizations", "locations", "dates", "keywords")}
    json_schema = {
        "type": "object",
        "properties": {k: {"type": "array", "items": {"type": "string"}} for k in merged},
        "required": list(merged.keys()),
    }

    for wi, window in enumerate(windows):
        if len(windows) > 1:
            _progress(ctx, "extracting", 80 + (wi * 4 // len(windows)), f"Encje okno {wi+1}/{len(windows)}…")
        prompt = (f"{lang_note}Przeanalizuj tekst i wyodrębnij encje. Odpowiedz WYŁĄCZNIE w JSON.\n\n"
                  f"Tekst:\n{window}\n\n"
                  "Zwróć JSON: persons, organizations, locations, dates, keywords (tablice stringów).")
        raw = _call_ollama_generate(prompt, json_format=json_schema, num_ctx=SUMMARY_NUM_CTX, timeout=120.0, model=ctx.ollama_model)
        if raw:
            try:
                data = json.loads(raw)
                for k in merged:
                    for v in data.get(k, []):
                        if v:
                            merged[k].add(v.strip())
            except (json.JSONDecodeError, AttributeError):
                pass

    ctx.entities = {k: sorted(v) for k, v in merged.items()}
    ctx.entities["keywords"] = ctx.entities["keywords"][:20]
    return {k: len(v) for k, v in ctx.entities.items()}


def _tool_build_rag_index(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")
    _progress(ctx, "embedding", 88, "Budowanie indeksu RAG…")

    CHUNK_CHARS, OVERLAP_CHARS, MAX_CHUNKS = 800, 150, 300
    texts: List[str] = []
    chunk_meta: List[Dict] = []
    buf = ""
    buf_seg_ids: List[int] = []
    buf_start = buf_end = buf_speaker = None

    full_seg_texts = [(s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments]
    for i, seg in enumerate(ctx.segments):
        txt = full_seg_texts[i]
        if not txt:
            continue
        if not buf:
            buf_start = seg.get("start")
            buf_speaker = seg.get("speaker")
        buf += (" " if buf else "") + txt
        buf_seg_ids.append(seg.get("id", i))
        buf_end = seg.get("end")
        if len(buf) >= CHUNK_CHARS:
            texts.append(buf.strip())
            chunk_meta.append({"segmentIds": buf_seg_ids[:], "start": buf_start, "end": buf_end, "speaker": buf_speaker})
            if len(texts) >= MAX_CHUNKS:
                break
            buf = buf[-OVERLAP_CHARS:]
            buf_seg_ids = buf_seg_ids[-3:]
            buf_start = buf_end

    if buf.strip() and len(texts) < MAX_CHUNKS:
        texts.append(buf.strip())
        chunk_meta.append({"segmentIds": buf_seg_ids[:], "start": buf_start, "end": buf_end, "speaker": buf_speaker})

    if not texts:
        return {"chunks": 0, "skipped": True}

    all_embeddings: List[List[float]] = []
    for b in range(0, len(texts), 32):
        vecs = _call_ollama_embed(texts[b: b + 32], timeout=120.0, embed_model=ctx.ollama_model)
        all_embeddings.extend(vecs if vecs else [[0.0] * 768] * len(texts[b: b + 32]))

    ctx.rag_entries = [
        {"id": i, "text": texts[i], "embedding": all_embeddings[i] if i < len(all_embeddings) else [], "metadata": chunk_meta[i]}
        for i in range(len(texts))
    ]
    return {"chunks": len(ctx.rag_entries), "embedding_model": OLLAMA_EMBEDDING_MODEL}


def _tool_summarize_transcript(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    if not ctx.segments:
        raise RuntimeError("No segments. Run transcribe_audio first.")

    style = args.get("style", "structured")
    full_text = " ".join((s.get("text_pl") or s.get("text") or "").strip() for s in ctx.segments)
    duration_min = round(ctx.duration / 60)
    is_long = ctx.duration > 30 * 60
    WINDOW_CHARS, OVERLAP_CHARS, MAX_WINDOWS = 6000, 600, 6

    lang_names = {"en": "angielski", "ru": "rosyjski", "uk": "ukraiński", "de": "niemiecki"}
    lang_note = (f" (język oryginalny: {lang_names.get(ctx.detected_language, ctx.detected_language)})"
                 if ctx.detected_language not in ("pl", "auto") else "")

    style_note = {
        "brief": "Napisz krótkie streszczenie (3-5 zdań).",
        "detailed": "Napisz szczegółowy raport.",
        "structured": ("Napisz raport:\n## Streszczenie\n(2-4 zdania)\n\n## Główne tematy\n(lista)\n\n"
                       "## Kluczowe decyzje i ustalenia\n(lista lub Brak)\n\n## Uczestnicy rozmowy\n"
                       "(imię/rola + język)\n\n## Następne kroki\n(lista lub Nie wspomniano)"),
    }.get(style, "Napisz szczegółowy raport.")

    def _summary_prompt(text: str) -> str:
        return f"Stwórz raport z transkrypcji rozmowy{lang_note}. PO POLSKU.\n\nTranskrypcja:\n{text}\n\n{style_note}\n\nRaport:"

    _progress(ctx, "summarizing", 93, f"Podsumowanie{' (długie nagranie)' if is_long else ''}…")

    if not is_long or len(full_text) <= WINDOW_CHARS:
        report = _call_ollama_generate(_summary_prompt(full_text[:WINDOW_CHARS]), num_ctx=SUMMARY_NUM_CTX, timeout=300.0, model=ctx.ollama_model)
    else:
        step = WINDOW_CHARS - OVERLAP_CHARS
        windows = [full_text[i: i + WINDOW_CHARS] for i in range(0, len(full_text), step)][:MAX_WINDOWS]
        partials: List[str] = []
        for wi, w in enumerate(windows):
            _progress(ctx, "summarizing", 93 + (wi * 3 // len(windows)), f"Streszczenie fragment {wi+1}/{len(windows)}…")
            p = _call_ollama_generate(_summary_prompt(w), num_ctx=SUMMARY_NUM_CTX, timeout=300.0, model=ctx.ollama_model)
            if p:
                partials.append(f"### Fragment {wi+1}\n{p}")
        if partials:
            _progress(ctx, "summarizing", 98, "Łączenie streszczeń…")
            combined = "\n\n".join(partials)[: WINDOW_CHARS * 2]
            reduce_prompt = (f"Częściowe streszczenia nagrania (~{duration_min} min). Jeden spójny raport PO POLSKU:\n\n"
                             f"{combined}\n\n## Streszczenie\n## Główne tematy\n## Uczestnicy\nRaport:")
            report = _call_ollama_generate(reduce_prompt, num_ctx=SUMMARY_NUM_CTX, timeout=360.0, model=ctx.ollama_model)
        else:
            report = ""

    if report:
        ctx.report = report
        ctx.summary = " ".join(report.split("\n")[:3])[:300]

    return {"report_length": len(report) if report else 0}


def _tool_run_analysis(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Code interpreter — executes Python analysis against transcript data."""
    code = (args.get("code") or "").strip()
    if not code:
        return {"error": "No code provided"}

    safe_locals: Dict[str, Any] = {
        "segments": ctx.segments,
        "speaker_profiles": ctx.speaker_profiles_raw,
        "display_names": ctx.display_names,
        "entities": ctx.entities,
        "quality_stats": ctx.quality_stats,
        "duration": ctx.duration,
        "detected_language": ctx.detected_language,
        "result": None,
    }
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            b: __builtins__[b] if isinstance(__builtins__, dict) else getattr(__builtins__, b, None)  # type: ignore[index]
            for b in ["len", "range", "enumerate", "zip", "map", "filter", "sorted",
                      "sum", "min", "max", "round", "abs", "list", "dict", "set",
                      "tuple", "str", "int", "float", "bool", "print", "isinstance",
                      "type", "any", "all", "iter", "next", "reversed"]
        },
        "json": json,
    }

    output_buf = io.StringIO()
    try:
        with redirect_stdout(output_buf):
            exec(code, safe_globals, safe_locals)  # noqa: S102
        output = output_buf.getvalue()
        result_val = safe_locals.get("result")
        return {
            "success": True,
            "output": output[:3000],
            "result": str(result_val)[:500] if result_val is not None else None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": output_buf.getvalue()[:500]}


def _tool_save_checkpoint(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Serialize current state to disk for context offloading."""
    label = args.get("label", "checkpoint")
    checkpoint_id = f"{ctx.session_id}_{int(time.time())}"
    path = CHECKPOINT_DIR / f"{checkpoint_id}.json"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclude audio PCM and large embeddings to keep file manageable
    state = {
        "checkpoint_id": checkpoint_id,
        "label": label,
        "session_id": ctx.session_id,
        "created_at": time.time(),
        "filename": ctx.filename,
        "duration": ctx.duration,
        "detected_language": ctx.detected_language,
        "asr_engine": ctx.asr_engine,
        "transcription_params": ctx.transcription_params,
        "segments": ctx.segments,          # text + timestamps only
        "speaker_profiles_raw": ctx.speaker_profiles_raw,
        "audio_features_raw": ctx.audio_features_raw,
        "display_names": ctx.display_names,
        "entities": ctx.entities,
        "summary": ctx.summary,
        "report": ctx.report,
        "quality_stats": ctx.quality_stats,
        # Skip rag_entries (large embeddings)
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        return {"checkpoint_id": checkpoint_id, "label": label, "segments_saved": len(ctx.segments)}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_load_checkpoint(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """Restore processing state from checkpoint."""
    checkpoint_id = args.get("checkpoint_id", "")
    path = CHECKPOINT_DIR / f"{checkpoint_id}.json"
    if not path.exists():
        return {"error": f"Checkpoint {checkpoint_id} not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        ctx.segments = state.get("segments", [])
        ctx.detected_language = state.get("detected_language", "auto")
        ctx.asr_engine = state.get("asr_engine", "whisper")
        ctx.transcription_params = state.get("transcription_params", {})
        ctx.speaker_profiles_raw = state.get("speaker_profiles_raw", {})
        ctx.audio_features_raw = state.get("audio_features_raw", {})
        ctx.display_names = state.get("display_names", {})
        ctx.entities = state.get("entities")
        ctx.summary = state.get("summary")
        ctx.report = state.get("report")
        ctx.quality_stats = state.get("quality_stats")
        return {"loaded": True, "segments": len(ctx.segments), "label": state.get("label")}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_save_memory(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    observation = args.get("observation", "").strip()
    improvement = args.get("improvement", "").strip()
    if not observation or not improvement:
        return {"error": "observation and improvement are required"}
    mem = save_memory(observation, improvement, args.get("tags", []))
    _agent_event(ctx, "agent_memory", memory=mem)
    return {"saved": True, "id": mem.get("id")}


def _tool_finish(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    return {
        "complete": True,
        "message": args.get("message", "Przetwarzanie zakończone."),
        "segment_count": len(ctx.segments),
        "detected_language": ctx.detected_language,
        "has_translation": any(s.get("text_pl") for s in ctx.segments),
        "has_entities": ctx.entities is not None,
        "has_summary": ctx.summary is not None,
        "rag_chunks": len(ctx.rag_entries),
    }


# ── New quality & analysis tools ──────────────────────────────────────────────

def _tool_refine_speaker_assignments(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Post-diarization speaker refinement using three strategies:
    1. Neighbour voting for short segments (<1.5s) — majority of ±4 neighbours wins.
    2. Micro-gap bridging — same speaker segments < gap_sec apart are merged.
    3. LLM-assisted ambiguity resolution — for segments where neighbours disagree,
       ask the LLM to infer speaker from textual cues (name mentions, pronouns).
    """
    if not ctx.segments:
        return {"error": "No segments. Run diarize_audio first."}

    from collections import Counter as _Counter

    gap_sec = float(args.get("gap_sec", 0.5))
    short_thresh = float(args.get("short_sec", 1.5))
    window = int(args.get("window", 4))

    changes = 0

    # Strategy 1: Neighbour majority vote for short segments
    for i, seg in enumerate(ctx.segments):
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur >= short_thresh:
            continue
        start_j = max(0, i - window)
        end_j = min(len(ctx.segments), i + window + 1)
        neighbours = [ctx.segments[j]["speaker"]
                      for j in range(start_j, end_j) if j != i]
        if not neighbours:
            continue
        top_sp, top_cnt = _Counter(neighbours).most_common(1)[0]
        if top_cnt >= max(2, len(neighbours) // 2) and top_sp != seg.get("speaker"):
            seg["speaker"] = top_sp
            changes += 1

    # Strategy 2: Bridge micro-gaps between same-speaker runs
    bridge_count = 0
    for i in range(1, len(ctx.segments) - 1):
        prev_sp = ctx.segments[i - 1].get("speaker")
        next_sp = ctx.segments[i + 1].get("speaker") if i + 1 < len(ctx.segments) else None
        curr_sp = ctx.segments[i].get("speaker")
        gap_before = (ctx.segments[i].get("start") or 0) - (ctx.segments[i - 1].get("end") or 0)
        gap_after = (ctx.segments[i + 1].get("start") if i + 1 < len(ctx.segments) else 999) - (ctx.segments[i].get("end") or 0)
        # If flanked by same speaker with tiny gaps, absorb
        if prev_sp == next_sp and prev_sp != curr_sp and gap_before < gap_sec and gap_after < gap_sec:
            ctx.segments[i]["speaker"] = prev_sp
            bridge_count += 1
            changes += 1

    # Strategy 3: LLM disambiguation for remaining ambiguous short segments
    ambiguous = []
    for i, seg in enumerate(ctx.segments):
        dur = (seg.get("end") or 0) - (seg.get("start") or 0)
        if dur < 0.8:
            text = (seg.get("text_pl") or seg.get("text") or "").strip()
            if text:
                # Build context window
                context_segs = ctx.segments[max(0, i-2):i] + ctx.segments[i+1:min(len(ctx.segments), i+3)]
                context_text = "\n".join(
                    f"[{s['speaker']}] {(s.get('text_pl') or s.get('text') or '').strip()}"
                    for s in context_segs
                )
                ambiguous.append((i, seg["speaker"], text, context_text))

    if ambiguous and len(ambiguous) <= 20:  # only for small batches
        prompt_lines = []
        for i, (idx, current_sp, text, context) in enumerate(ambiguous):
            speakers = list({s.get("speaker") for s in ctx.segments if s.get("speaker")})
            prompt_lines.append(
                f"{i+1}. Krótki segment [{current_sp}]: \"{text}\"\n"
                f"   Kontekst:\n{context}\n"
                f"   Dostępni mówcy: {speakers}"
            )
        prompt = (
            "Który mówca powiedział poniższe krótkie segmenty? "
            "Skorzystaj z kontekstu (nazwy, tytuły, zaimki, zwroty).\n\n"
            + "\n\n".join(prompt_lines)
            + "\n\nZwróć JSON: {\"assignments\": [{\"index\": int, \"speaker\": str}, ...]}"
        )
        schema = {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "speaker": {"type": "string"},
                        },
                        "required": ["index", "speaker"],
                    },
                }
            },
            "required": ["assignments"],
        }
        try:
            raw = _call_ollama_generate(prompt, json_format=schema, num_ctx=8192, timeout=60.0, model=ctx.ollama_model)
            if raw:
                data = json.loads(raw)
                valid_speakers = {s.get("speaker") for s in ctx.segments}
                for item in data.get("assignments", []):
                    local_idx = int(item.get("index", -1)) - 1
                    new_sp = item.get("speaker", "")
                    if 0 <= local_idx < len(ambiguous) and new_sp in valid_speakers:
                        seg_idx = ambiguous[local_idx][0]
                        if ctx.segments[seg_idx]["speaker"] != new_sp:
                            ctx.segments[seg_idx]["speaker"] = new_sp
                            changes += 1
        except Exception as exc:
            logger.debug("LLM speaker disambiguation failed (non-fatal): %s", exc)

    return {
        "changes": changes,
        "short_segments_checked": len([s for s in ctx.segments if (s.get("end", 0) - s.get("start", 0)) < short_thresh]),
        "bridges": bridge_count,
        "llm_disambiguated": len(ambiguous),
    }


def _tool_extract_quotes_and_facts(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Extract structured quotes, facts, decisions, and key questions from the transcript.
    Results stored in ctx and included in the final result event.
    Critical for analysis quality — quotes provide verbatim evidence.
    """
    if not ctx.segments:
        return {"error": "No segments. Run transcribe_audio first."}

    def fmt_time(s: float) -> str:
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m}:{sec:02d}"

    # Build labelled transcript with timestamps
    lines: List[str] = []
    for seg in ctx.segments:
        text = (seg.get("text_pl") or seg.get("text") or "").strip()
        if not text:
            continue
        sp = seg.get("speaker", "?")
        display = ctx.display_names.get(sp, sp)
        t = fmt_time(float(seg.get("start") or 0))
        lines.append(f"[{t} {display}] {text}")

    transcript = "\n".join(lines)[:8000]

    prompt = (
        "Jesteś ekspertem od analizy dyskursu. Przeanalizuj poniższy transkrypt i wyodrębnij:\n\n"
        "1. CYTATY — dosłowne, znaczące wypowiedzi (min. 5, max. 15). "
        "Priorytet: emocjonalne, kontrowersyjne, kluczowe dla tematu.\n"
        "2. FAKTY — konkretne twierdzenia, liczby, daty, nazwy.\n"
        "3. DECYZJE — co zostało postanowione, uzgodnione.\n"
        "4. KLUCZOWE PYTANIA — ważne pytania zadane w rozmowie.\n\n"
        f"Transkrypt:\n{transcript}\n\n"
        "Format JSON:\n"
        "{\n"
        "  \"quotes\": [{\"speaker\": str, \"timestamp\": str, \"text\": str, \"significance\": str}],\n"
        "  \"facts\": [{\"speaker\": str, \"text\": str, \"category\": \"number|date|name|claim\"}],\n"
        "  \"decisions\": [{\"text\": str, \"participants\": [str]}],\n"
        "  \"key_questions\": [{\"speaker\": str, \"text\": str}]\n"
        "}"
    )
    schema = {
        "type": "object",
        "properties": {
            "quotes": {"type": "array", "items": {"type": "object"}},
            "facts": {"type": "array", "items": {"type": "object"}},
            "decisions": {"type": "array", "items": {"type": "object"}},
            "key_questions": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["quotes", "facts"],
    }

    _progress(ctx, "extracting", 85, "Ekstrakcja cytatów i faktów…")
    raw = _call_ollama_generate(prompt, json_format=schema, num_ctx=SUMMARY_NUM_CTX, timeout=180.0, model=ctx.ollama_model)

    if not raw:
        return {"error": "LLM did not respond"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse quotes/facts response"}

    # Store in context
    ctx._quotes_and_facts = data  # type: ignore[attr-defined]

    # Inject quotes as high-priority RAG entries (prepend to rag_entries)
    for i, q in enumerate(data.get("quotes", [])[:15]):
        text = q.get("text", "")
        speaker = q.get("speaker", "")
        ts = q.get("timestamp", "")
        if text:
            # Find matching segment for timestamps
            ctx.rag_entries.insert(i, {
                "id": -(i + 1),  # negative IDs mark special entries
                "text": f"CYTAT [{ts} {speaker}]: {text}",
                "embedding": [],  # will be recomputed if build_rag_index is called again
                "metadata": {"type": "quote", "speaker": speaker, "timestamp": ts},
            })

    return {
        "quotes_extracted": len(data.get("quotes", [])),
        "facts_extracted": len(data.get("facts", [])),
        "decisions_extracted": len(data.get("decisions", [])),
        "key_questions_extracted": len(data.get("key_questions", [])),
        "quotes_sample": [q.get("text", "")[:80] for q in data.get("quotes", [])[:3]],
    }


def _tool_verify_names_and_locations(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    LLM-powered verification of identified names and locations.
    For each entity that looks uncertain (phonetically similar alternatives,
    context mismatch), provides:
    - canonical form
    - alternatives with likelihood scores
    - context evidence
    Updates ctx.entities with verified versions.
    """
    if not ctx.entities:
        return {"skipped": True, "reason": "No entities. Run extract_entities first."}

    persons = ctx.entities.get("persons", [])
    locations = ctx.entities.get("locations", [])

    if not persons and not locations:
        return {"skipped": True, "reason": "No persons or locations found"}

    # Build context evidence from transcript
    context_lines: List[str] = []
    for seg in ctx.segments[:40]:
        text = (seg.get("text_pl") or seg.get("text") or "").strip()
        if text:
            context_lines.append(f"[{seg.get('speaker','?')}] {text}")
    context_sample = "\n".join(context_lines[:30])

    prompt = (
        "Jesteś ekspertem lingwistycznym. Zweryfikuj poniższe nazwy własne wyodrębnione z transkrypcji.\n"
        "Sprawdź:\n"
        "1. Czy pisownia jest poprawna w kontekście rozmowy?\n"
        "2. Czy istnieją fonetycznie podobne alternatywy (np. 'Kasia' → 'Katarzyna', 'Wrocław' → 'Wrocław OK')?\n"
        "3. Oceń pewność 0.0-1.0.\n\n"
        f"Osoby: {json.dumps(persons, ensure_ascii=False)}\n"
        f"Miejsca: {json.dumps(locations, ensure_ascii=False)}\n\n"
        f"Fragment transkryptu (kontekst):\n{context_sample[:3000]}\n\n"
        "Zwróć JSON:\n"
        "{\n"
        "  \"verified_persons\": [{\"original\": str, \"canonical\": str, \"alternatives\": [str], \"confidence\": float, \"note\": str}],\n"
        "  \"verified_locations\": [{\"original\": str, \"canonical\": str, \"alternatives\": [str], \"confidence\": float}]\n"
        "}"
    )
    schema = {
        "type": "object",
        "properties": {
            "verified_persons": {"type": "array", "items": {"type": "object"}},
            "verified_locations": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["verified_persons", "verified_locations"],
    }

    _progress(ctx, "extracting", 87, "Weryfikacja nazw i lokalizacji…")
    raw = _call_ollama_generate(prompt, json_format=schema, num_ctx=SUMMARY_NUM_CTX, timeout=120.0, model=ctx.ollama_model)

    if not raw:
        return {"error": "LLM did not respond"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Parse failed"}

    # Update entities with canonical forms
    verified_persons = data.get("verified_persons", [])
    verified_locations = data.get("verified_locations", [])

    if verified_persons:
        canonical = [v.get("canonical") or v.get("original", "") for v in verified_persons if v.get("canonical")]
        ctx.entities["persons"] = canonical

    if verified_locations:
        canonical_locs = [v.get("canonical") or v.get("original", "") for v in verified_locations if v.get("canonical")]
        ctx.entities["locations"] = canonical_locs

    # Store full verification report
    ctx._entity_verification = data  # type: ignore[attr-defined]

    uncertain = [v for v in verified_persons + verified_locations if float(v.get("confidence", 1.0)) < 0.7]

    return {
        "persons_verified": len(verified_persons),
        "locations_verified": len(verified_locations),
        "uncertain_count": len(uncertain),
        "uncertain": [{"name": u.get("original"), "alternatives": u.get("alternatives", []), "confidence": u.get("confidence")} for u in uncertain[:5]],
    }


def _tool_multi_pass_transcribe_segment(args: Dict, ctx: AgentContext, models: Dict) -> Dict:
    """
    Re-transcribe a specific segment with a context-primed prompt.
    Use when verify_transcript_quality reports low confidence for a specific region.
    Provides the LLM with surrounding text as initial_prompt to disambiguate.
    """
    if ctx.audio_pcm is None or not ctx.segments:
        return {"error": "Audio and segments required."}

    segment_id = int(args.get("segment_id", -1))
    if segment_id < 0 or segment_id >= len(ctx.segments):
        return {"error": f"Invalid segment_id {segment_id}"}

    whisper = models.get("whisper")
    if whisper is None:
        return {"error": "Whisper not available"}
    models.get("ensure_whisper_gpu", lambda: None)()

    seg = ctx.segments[segment_id]
    start_sec = float(seg.get("start") or 0)
    end_sec = float(seg.get("end") or 0)
    padding = float(args.get("padding_sec", 2.0))  # extra context
    start_smp = int(max(0, start_sec - padding) * ctx.sample_rate)
    end_smp = int(min(len(ctx.audio_pcm), (end_sec + padding) * ctx.sample_rate))
    fragment = ctx.audio_pcm[start_smp:end_smp]

    # Build initial_prompt from the 2 preceding segments
    preceding = [ctx.segments[i] for i in range(max(0, segment_id - 2), segment_id)]
    initial_prompt = " ".join((s.get("text") or "").strip() for s in preceding)[:120] or None

    try:
        chunks, _, _ = whisper.transcribe(
            fragment,
            ctx.language_hint or ctx.detected_language,
            progress_cb=lambda *a: None,
            extra_kw={
                "beam_size": 8,
                "best_of": 8,
                "temperature": 0.0,
                "vad_filter": False,  # don't filter — segment is pre-selected
                **({"initial_prompt": initial_prompt} if initial_prompt else {}),
            },
        )
    except Exception as exc:
        return {"error": str(exc)}

    # Find the chunk that best overlaps with our target segment
    best_chunk = None
    best_overlap = 0.0
    for ch in chunks:
        ch_start = ch.get("start", 0) + max(0, start_sec - padding)
        ch_end = ch.get("end", 0) + max(0, start_sec - padding)
        overlap = min(ch_end, end_sec) - max(ch_start, start_sec)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = ch

    if not best_chunk:
        return {"no_improvement": True, "note": "No overlapping chunk found"}

    new_text = best_chunk.get("text", "").strip()
    old_text = seg.get("text", "")
    new_conf = round(
        sum(float(w.get("probability", 1.0)) for w in (best_chunk.get("words") or [])) /
        max(1, len(best_chunk.get("words") or [])), 3
    )
    old_conf = ctx.segment_quality.get(segment_id, 1.0)

    if new_conf > old_conf and new_text:
        ctx.segments[segment_id]["text"] = new_text
        ctx.segments[segment_id]["text_pl"] = None  # mark for re-translation
        ctx.segments[segment_id]["words"] = best_chunk.get("words", [])
        ctx.segment_quality[segment_id] = new_conf
        _send(ctx, {"type": "segment_chunk", "segments": [{
            "id": segment_id,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": new_text,
            "speaker": seg.get("speaker", "—"),
        }], "offset_sec": start_sec, "cumulative": len(ctx.segments)})
        return {"improved": True, "old_text": old_text[:80], "new_text": new_text[:80],
                "old_confidence": old_conf, "new_confidence": new_conf}
    else:
        return {"improved": False, "note": "No improvement found", "old_conf": old_conf, "new_conf": new_conf}


# ── Tool dispatch table ───────────────────────────────────────────────────────

_TOOL_IMPL = {
    # Probing & calibration
    "probe_audio_fragment": _tool_probe_audio_fragment,
    "detect_speaker_count": _tool_detect_speaker_count,
    "set_transcription_params": _tool_set_transcription_params,
    "analyze_audio_quality": _tool_analyze_audio_quality,
    "detect_language_switches": _tool_detect_language_switches,
    # Core processing
    "get_audio_info": _tool_get_audio_info,
    "transcribe_audio": _tool_transcribe_audio,
    "verify_transcript_quality": _tool_verify_transcript_quality,
    "split_long_segments": _tool_split_long_segments,
    "merge_short_segments": _tool_merge_short_segments,
    "diarize_audio": _tool_diarize_audio,
    "normalize_speaker_labels": _tool_normalize_speaker_labels,
    "profile_speakers": _tool_profile_speakers,
    # Translation (sub-agent)
    "translate_to_polish": _tool_translate_to_polish,
    "validate_translation_quality": _tool_validate_translation_quality,
    "retranslate_segments": _tool_retranslate_segments,
    # Speakers & entities
    "identify_speakers": _tool_identify_speakers,
    "extract_entities": _tool_extract_entities,
    "extract_keywords_statistical": _tool_extract_keywords_statistical,
    "detect_topics": _tool_detect_topics,
    # Text analysis
    "compute_text_statistics": _tool_compute_text_statistics,
    "search_in_transcript": _tool_search_in_transcript,
    # Indexing & synthesis
    "build_rag_index": _tool_build_rag_index,
    "summarize_transcript": _tool_summarize_transcript,
    # UI & context
    "emit_partial_result": _tool_emit_partial_result,
    "run_analysis": _tool_run_analysis,
    "save_checkpoint": _tool_save_checkpoint,
    "load_checkpoint": _tool_load_checkpoint,
    # Memory & control
    "save_memory": _tool_save_memory,
    "finish": _tool_finish,
    # New experiment + artifacts
    "compare_transcription_params": _tool_compare_transcription_params,
    "retranscribe_time_range": _tool_retranscribe_time_range,
    "write_artifact": _tool_write_artifact,
    "read_artifact": _tool_read_artifact,
    "list_artifacts": _tool_list_artifacts,
    # Quality & analysis
    "refine_speaker_assignments": _tool_refine_speaker_assignments,
    "extract_quotes_and_facts": _tool_extract_quotes_and_facts,
    "verify_names_and_locations": _tool_verify_names_and_locations,
    "multi_pass_transcribe_segment": _tool_multi_pass_transcribe_segment,
}


# ── Result builders ───────────────────────────────────────────────────────────

def _build_speaker_profiles(ctx: AgentContext) -> Dict:
    return {
        sp: {
            "gender": p.get("gender"),
            "gender_probs": p.get("gender_probs"),
            "age_estimate": p.get("age_estimate"),
            "age_group": p.get("age_group"),
            "confidence": p.get("confidence"),
            "display_name": ctx.display_names.get(sp),
            **ctx.audio_features_raw.get(sp, {}),
        }
        for sp, p in ctx.speaker_profiles_raw.items()
    }


def _build_segments_out(ctx: AgentContext) -> List[Dict]:
    return [
        {
            "id": i,
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text", ""),
            "text_pl": s.get("text_pl") or s.get("text", ""),
            "speaker": s.get("speaker", f"GŁOS_{i+1:02d}"),
            "language": s.get("language") or ctx.detected_language,
            "words": s.get("words") or [],
        }
        for i, s in enumerate(ctx.segments)
    ]


# ── Main agent loop ───────────────────────────────────────────────────────────

def _build_system_prompt(ctx: AgentContext) -> str:
    memories_block = format_memories_for_prompt(load_memories())
    return (
        "You are an expert audio intelligence agent for Pandaro.\n"
        "Model: {model} | Session: {sid}\n\n"
        "## MISSION\n"
        "Produce an excellent transcript with accurate speaker attribution and rich analysis.\n"
        "Experiment, validate, self-correct. Use all available tools.\n\n"
        "## TOOLS\n"
        "CALIBRATION:    get_audio_info · analyze_audio_quality · probe_audio_fragment\n"
        "                compare_transcription_params · set_transcription_params\n"
        "                detect_speaker_count · detect_language_switches\n"
        "TRANSCRIPTION:  transcribe_audio (streams live) · verify_transcript_quality\n"
        "                multi_pass_transcribe_segment · split_long_segments · merge_short_segments\n"
        "SPEAKERS:       diarize_audio · refine_speaker_assignments · normalize_speaker_labels\n"
        "                profile_speakers · identify_speakers\n"
        "TRANSLATION:    translate_to_polish (sub-agent with quality checks)\n"
        "                validate_translation_quality · retranslate_segments\n"
        "ANALYSIS:       extract_entities · verify_names_and_locations\n"
        "                extract_quotes_and_facts · extract_keywords_statistical\n"
        "                detect_topics · compute_text_statistics · run_analysis\n"
        "SYNTHESIS:      summarize_transcript · build_rag_index\n"
        "UI/CONTEXT:     emit_partial_result · search_in_transcript\n"
        "                save_checkpoint · load_checkpoint\n"
        "ARTIFACTS:      write_artifact · read_artifact · list_artifacts\n"
        "MEMORY:         save_memory · finish\n\n"
        "## QUALITY LOOP (MANDATORY)\n"
        "### Transcription quality loop:\n"
        "1. analyze_audio_quality → set VAD params\n"
        "2. compare_transcription_params on 60s sample → find best params\n"
        "3. set_transcription_params → transcribe_audio\n"
        "4. verify_transcript_quality → IF avg_confidence < 0.70:\n"
        "   a) For low-confidence segments: multi_pass_transcribe_segment (up to 5 worst)\n"
        "   b) OR retranscribe_time_range for a whole bad section\n"
        "   c) Retry transcribe_audio with adjusted params (once max)\n\n"
        "### Diarization quality loop:\n"
        "5. detect_speaker_count → diarize_audio(num_speakers=N)\n"
        "6. merge_short_segments → refine_speaker_assignments\n"
        "   ↳ refine_speaker_assignments uses neighbour voting + LLM for ambiguous short segments\n"
        "7. compute_text_statistics → IF one speaker > 85%: re-run diarize with different num_speakers\n"
        "8. profile_speakers → identify_speakers\n\n"
        "### Translation quality loop:\n"
        "9. translate_to_polish (has built-in quality checks per batch)\n"
        "10. validate_translation_quality → IF avg_score < 3.5: retranslate_segments(worst_ids)\n\n"
        "### Analysis (MANDATORY — DO NOT SKIP):\n"
        "11. emit_partial_result — show user live preview\n"
        "12. extract_entities → verify_names_and_locations\n"
        "    ↳ Pay SPECIAL ATTENTION to names and locations:\n"
        "    - Provide alternatives for phonetically similar names\n"
        "    - Check if locations make geographic sense in context\n"
        "    - For audio recorded in Russia/Ukraine: verify Cyrillic transliterations\n"
        "13. extract_quotes_and_facts — MANDATORY. Verbatim quotes are critical for Q&A quality.\n"
        "    ↳ Quotes are injected as high-priority RAG entries automatically.\n"
        "14. detect_topics (for recordings > 10 min)\n"
        "15. build_rag_index (embeddings for semantic search)\n"
        "16. summarize_transcript(style='structured')\n"
        "17. save_memory — record patterns useful for future sessions\n"
        "18. finish\n\n"
        "## QUALITY TARGETS\n"
        "☑ avg word confidence >= 0.70\n"
        "☑ all segments have correct speaker labels (use refine_speaker_assignments)\n"
        "☑ text_pl on every segment\n"
        "☑ names/locations verified with alternatives\n"
        "☑ QUOTES extracted (minimum 5 key quotes)\n"
        "☑ entities, topics, summary, RAG index\n\n"
        "## SPECIAL INSTRUCTIONS\n"
        "- Short segments (<1s) are prone to mis-assignment — always refine_speaker_assignments\n"
        "- For multi-speaker recordings: verify speaker balance via compute_text_statistics\n"
        "- Use write_artifact to save intermediate analysis notes\n"
        "- If Ollama is slow: save_checkpoint before heavy LLM operations\n\n"
        "## VRAM\n"
        "whisper=6GB · vibevoice=18GB (built-in diarization) · nemotron=1.2GB\n"
    ).format(model=ctx.ollama_model, sid=ctx.session_id) + memories_block


def run_agent(ctx: AgentContext, models: Dict) -> None:
    """Main agent loop. Runs synchronously in the GPU thread executor."""
    _agent_event(ctx, "agent_start", message="Agent uruchomiony.", session_id=ctx.session_id)
    register_session(ctx)

    # Cache check — key includes model so changing model invalidates cache
    transcribe_cache = models.get("transcribe_cache")
    if transcribe_cache and ctx.filename:
        import hashlib
        sha = hashlib.sha256(ctx.audio_content).hexdigest()
        cache_key = transcribe_cache.key(sha, ctx.language_hint, ctx.ollama_model, ctx.do_translate)
        cached = transcribe_cache.get(cache_key)
        if cached is not None:
            _progress(ctx, "done", 100, "Wynik z cache.")
            _send(ctx, {**cached, "cached": True})
            asyncio.run_coroutine_threadsafe(ctx.queue.put(None), ctx.loop)
            deregister_session(ctx.session_id)
            return
        ctx.audio_sha = sha
        ctx._cache_key = cache_key  # type: ignore[attr-defined]

    messages: List[Dict] = [
        {"role": "system", "content": _build_system_prompt(ctx)},
        {
            "role": "user",
            "content": (
                f"Process audio file: '{ctx.filename}'. "
                f"Language hint: '{ctx.language_hint or 'auto'}'. "
                f"Translate to Polish: {ctx.do_translate}. "
                "Follow the workflow: start with get_audio_info, probe the fragment, set params if needed, then proceed."
            ),
        },
    ]

    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        ctx.current_step = step
        _agent_event(ctx, "agent_thinking", step=step)

        # Inject any pending human hints as user messages
        while ctx.pending_hints:
            hint = ctx.pending_hints.popleft()
            messages.append({"role": "user", "content": f"[HUMAN HINT] {hint}"})
            _agent_event(ctx, "hint_injected", hint=hint, step=step)

        message = _call_ollama_chat(messages, tools=TOOL_SCHEMAS, num_ctx=AGENT_NUM_CTX, model=ctx.ollama_model)

        if message is None:
            logger.error("Ollama returned no message on step %d — running fallback", step)
            _run_fallback_pipeline(ctx, models)
            done = True
            break

        assistant_msg = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
        }
        messages.append(assistant_msg)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = (message.get("content") or "").lower()
            if any(w in content for w in ("complete", "finish", "done", "gotowe", "zakończono")) or step >= MAX_STEPS - 2:
                _finalize(ctx, models)
                done = True
            continue

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            raw_args = fn.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            attempts = ctx.tool_attempts.get(tool_name, 0)
            if attempts >= MAX_RETRIES_PER_TOOL:
                result: Dict = {"error": f"Tool {tool_name} exceeded retry limit.", "skipped": True}
                _agent_event(ctx, "tool_error", tool=tool_name, error="retry limit exceeded", skipped=True)
            else:
                ctx.tool_attempts[tool_name] = attempts + 1
                _agent_event(ctx, "tool_call", tool=tool_name, args=raw_args, attempt=attempts + 1)
                try:
                    impl = _TOOL_IMPL.get(tool_name)
                    if impl is None:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    result = impl(raw_args, ctx, models)
                    _agent_event(ctx, "tool_result", tool=tool_name, success=True, result=result)
                except Exception as exc:
                    logger.exception("Tool %s failed: %s", tool_name, exc)
                    result = {"error": str(exc)}
                    _agent_event(ctx, "tool_error", tool=tool_name, error=str(exc), attempt=attempts + 1)

            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

            if tool_name == "finish":
                done = True
                _finalize(ctx, models)
                break

    if not done:
        _finalize(ctx, models)

    deregister_session(ctx.session_id)
    asyncio.run_coroutine_threadsafe(ctx.queue.put(None), ctx.loop)


def _finalize(ctx: AgentContext, models: Dict) -> None:
    if not ctx.segments:
        _send(ctx, {"type": "error", "message": "Agent zakończył bez transkrypcji."})
        return

    result_event = {
        "type": "result",
        "segments": _build_segments_out(ctx),
        "detected_language": ctx.detected_language,
        "duration": round(ctx.duration, 2),
        "speaker_profiles": _build_speaker_profiles(ctx),
        "entities": ctx.entities,
        "summary": ctx.summary,
        "report": ctx.report,
        "rag_entries": ctx.rag_entries,
        "quality_stats": ctx.quality_stats,
        "segment_quality": ctx.segment_quality,
        "topics": ctx.topics,
        "quotes_and_facts": getattr(ctx, "_quotes_and_facts", None),
        "entity_verification": getattr(ctx, "_entity_verification", None),
        "asr_engine": ctx.asr_engine,
        "model_used": ctx.ollama_model,
    }

    transcribe_cache = models.get("transcribe_cache")
    cache_key = getattr(ctx, "_cache_key", None)
    if transcribe_cache and cache_key:
        transcribe_cache.put(cache_key, result_event)

    _progress(ctx, "done", 100, "Gotowe!")
    _send(ctx, result_event)

    # Release the model from Ollama GPU memory after session completes
    _offload_ollama_model(ctx.ollama_model)


def _run_fallback_pipeline(ctx: AgentContext, models: Dict) -> None:
    """Linear fallback pipeline when Ollama is unavailable."""
    logger.warning("Ollama unavailable — running fallback linear pipeline")
    _agent_event(ctx, "agent_thinking", message="Tryb awaryjny (Ollama niedostępne).")
    try:
        _tool_get_audio_info({}, ctx, models)
        _tool_transcribe_audio({"engine": "whisper", "language": ctx.language_hint or "auto"}, ctx, models)
        _tool_diarize_audio({}, ctx, models)
        _tool_merge_short_segments({}, ctx, models)
        _tool_profile_speakers({}, ctx, models)
        if ctx.do_translate:
            _tool_translate_to_polish({}, ctx, models)
        else:
            for s in ctx.segments:
                s["text_pl"] = s["text"]
        _tool_identify_speakers({}, ctx, models)
        _tool_emit_partial_result({"message": "Podgląd transkryptu"}, ctx, models)
        _tool_extract_entities({}, ctx, models)
        _tool_build_rag_index({}, ctx, models)
        _tool_summarize_transcript({"style": "structured"}, ctx, models)
    except Exception as exc:
        logger.exception("Fallback pipeline error: %s", exc)
        _send(ctx, {"type": "error", "message": str(exc)})
        return
    _finalize(ctx, models)

