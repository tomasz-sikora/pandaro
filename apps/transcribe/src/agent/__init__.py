"""
Pandaro audio intelligence agent — package root.

Public API:
  run_agent(ctx, models) — main agent loop
  AgentContext — session context dataclass
  inject_hint / get_active_sessions / register_session / deregister_session
"""
from __future__ import annotations
import asyncio, json, logging, time
from typing import Any, Dict, List, Optional
import numpy as np

from .context import (
    AgentContext, register_session, deregister_session,
    inject_hint, cancel_session, is_busy, get_active_sessions, _active_sessions,
)
from .helpers import (
    _send, _progress, _agent_event, _offload_ollama_model,
    _call_ollama_chat, AGENT_NUM_CTX, MAX_STEPS,
)
from .tools import _TOOL_IMPL
from .result_builders import _build_speaker_profiles, _build_segments_out
from .prompts import _build_system_prompt, TOOL_SCHEMAS
from ..memory import save_memory  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

# Import fallback tool functions for the linear pipeline
from .tools import (
    _tool_get_audio_info, _tool_transcribe_audio, _tool_diarize_audio,
    _tool_merge_short_segments, _tool_profile_speakers, _tool_translate_to_polish,
    _tool_identify_speakers, _tool_emit_partial_result, _tool_extract_entities,
    _tool_build_rag_index, _tool_summarize_transcript,
)
from .helpers import MAX_RETRIES_PER_TOOL

# ── Tool name aliases (handle LLM hallucinations / truncated names) ──
_TOOL_ALIASES: Dict[str, str] = {
    "diarize_first":              "diarize_first_transcribe",
    "diarize_and_transcribe":     "diarize_first_transcribe",
    "transcribe_by_turns":        "diarize_first_transcribe",
    "speaker_first_transcribe":   "diarize_first_transcribe",
    "refine_speaker_segments":    "refine_speaker_assignments",
    "refine_speakers":            "refine_speaker_assignments",
    "assign_speakers":            "refine_speaker_assignments",
    "merge_speakers":             "merge_duplicate_speakers",
    "deduplicate_speakers":       "merge_duplicate_speakers",
    "diarize":                    "diarize_audio",
    "transcribe":                 "transcribe_audio",
    "translate":                  "translate_to_polish",
    "summarize":                  "summarize_transcript",
    "entities":                   "extract_entities",
    "build_rag":                  "build_rag_index",
    "detect_noise":               "detect_noise_regions",
    "tag":                        "tag_segments",
    "check_quality":              "verify_transcript_quality",
    "quality_check":              "verify_transcript_quality",
}

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
            # Session-specific info lives here (NOT in the system prompt) so
            # the system prompt prefix is identical across sessions — Ollama
            # reuses the evaluated KV states and skips re-prefill every time.
            "content": (
                f"[Session: {ctx.session_id} | Model: {ctx.ollama_model} | "
                f"max_ctx_segments: {ctx.max_ctx_segments}]\n\n"
                f"Audio file: '{ctx.filename}' | "
                f"Language hint: '{ctx.language_hint or 'auto'}' | "
                f"Translate to Polish: {ctx.do_translate}.\n\n"
                "Call ONE tool per response. Start with get_audio_info."
            ),
        },
    ]

    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        ctx.current_step = step

        # ── Cancellation check (between steps) ────────────────────────────
        if ctx.cancelled:
            logger.info("Session %s cancelled at step %d", ctx.session_id, step)
            _send(ctx, {"type": "cancelled", "message": "Przetwarzanie anulowane przez użytkownika.",
                        "step": step})
            done = True
            break

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
                # Resolve aliases (handle hallucinated/truncated tool names)
                canonical_name = _TOOL_ALIASES.get(tool_name, tool_name)
                if canonical_name != tool_name:
                    logger.info("Tool alias: %s → %s", tool_name, canonical_name)
                _agent_event(ctx, "tool_call", tool=canonical_name, args=raw_args, attempt=attempts + 1,
                             original_name=tool_name if canonical_name != tool_name else None)
                try:
                    impl = _TOOL_IMPL.get(canonical_name)
                    if impl is None:
                        # Fuzzy match: find closest tool name for helpful error
                        import difflib
                        close = difflib.get_close_matches(tool_name, list(_TOOL_IMPL.keys()), n=3, cutoff=0.4)
                        raise ValueError(
                            f"Unknown tool: '{tool_name}'. "
                            + (f"Did you mean: {', '.join(close)}?" if close else "Check available tools.")
                        )
                    result = impl(raw_args, ctx, models)
                    _agent_event(ctx, "tool_result", tool=canonical_name, success=True, result=result)
                except Exception as exc:
                    logger.exception("Tool %s failed: %s", canonical_name, exc)
                    result = {"error": str(exc)}
                    _agent_event(ctx, "tool_error", tool=canonical_name, error=str(exc), attempt=attempts + 1)

            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

            if canonical_name == "finish" or tool_name == "finish":
                done = True
                _finalize(ctx, models)
                break

            # Abort the tool-call batch if cancelled mid-way
            if ctx.cancelled:
                break

    if not done and not ctx.cancelled:
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
        "asr_engine": ctx.asr_engine,
        "model_used": ctx.ollama_model,
    }

    transcribe_cache = models.get("transcribe_cache")
    cache_key = getattr(ctx, "_cache_key", None)
    if transcribe_cache and cache_key:
        transcribe_cache.put(cache_key, result_event)

    _progress(ctx, "done", 100, "Gotowe!")
    _send(ctx, result_event)

    # INTENTIONALLY do NOT offload the model here.
    # _offload_ollama_model(keep_alive=0) would unload 26 GB from VRAM so the
    # next upload has to reload it (~90 s penalty).  keep_alive=-1 keeps it hot
    # and Ollama will evict it automatically if another model needs the VRAM.
    # To force offload after idle, set OLLAMA_KEEP_ALIVE env on the Ollama side.


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



from .reprocess import run_reprocess  # noqa: E402

__all__ = [
    "AgentContext", "run_agent", "run_reprocess",
    "register_session", "deregister_session",
    "inject_hint", "cancel_session", "is_busy", "get_active_sessions",
]
