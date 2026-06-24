"""
FastAPI transcription service — agent-based architecture (v3).

POST /transcribe
  multipart/form-data: file, language?, translate?, engine?
  Returns text/event-stream → agent events + final result.

GET  /health            → engines status, diarizer, profiler, ollama
GET  /engines           → list of available (loaded) engines
GET  /memories          → list agent skill memories
DELETE /memories/{id}   → delete a memory by id
DELETE /memories        → clear all memories
GET  /cache/info        → LRU cache statistics

VRAM strategy (unchanged):
  - Whisper loaded at startup (~6 GB).
  - VibeVoice (9B, ~18 GB) loaded lazily, unloaded after use.
  - Agent (gemma4:26b) decides engine choice per session.
"""
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .transcriber import WhisperTranscriber
from .vibevoice_transcriber import VibeVoiceTranscriber
from .diarizer import Diarizer
from .speaker_profiler import SpeakerProfiler
from .audio_features import AudioFeatureExtractor
from .translator import ollama_available
from .cache import LRUCache
from .agent import (
    AgentContext, run_agent, run_reprocess, inject_hint, cancel_session,
    is_busy, get_active_sessions,
)
from .memory import list_memories, delete_memory, clear_all_memories
from . import translator as _translator_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Help PyTorch avoid VRAM fragmentation when models are swapped
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = FastAPI(title="pandaro Transcription Service", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-GPU sequential executor — ensures model swaps don't race
_executor = ThreadPoolExecutor(max_workers=1)

# In-memory LRU caches (keyed by request parameters, never embeddings)
_transcribe_cache: LRUCache = LRUCache(maxsize=10, name="transcribe")

# Whisper loaded at startup; VibeVoice loaded lazily per first use
_whisper: Optional[WhisperTranscriber] = None
_vibevoice: Optional[VibeVoiceTranscriber] = None   # None = not yet loaded; loaded on first VV request
_vibevoice_available: bool = False                   # True = imports OK, can be loaded
_diarizer: Optional[Diarizer] = None
_profiler: Optional[SpeakerProfiler] = None
_audio_features: Optional[AudioFeatureExtractor] = None
# Track which engine currently holds GPU memory
_gpu_engine: Optional[str] = None  # "whisper" | "vibevoice" | None


@app.on_event("startup")
async def startup():
    global _whisper, _vibevoice_available, _diarizer, _profiler, _audio_features, _gpu_engine
    loop = asyncio.get_event_loop()
    logger.info("Loading Whisper at startup (VibeVoice loads lazily on first use)…")
    result = await loop.run_in_executor(_executor, _load_startup_models)
    _whisper, _vibevoice_available, _diarizer, _profiler, _audio_features, _gpu_engine = result
    logger.info("Startup models loaded — agent-based pipeline active.")


def _load_startup_models():
    whisper = None
    gpu_engine = None
    try:
        whisper = WhisperTranscriber()
        gpu_engine = "whisper"
        logger.info("Whisper loaded on GPU.")
    except Exception as e:
        logger.error(f"Whisper load failed: {e}")

    # Check VibeVoice importability without loading weights
    vv_available = False
    try:
        from .vibevoice_transcriber import VibeVoiceTranscriber as _VV  # noqa: F401
        vv_available = True
        logger.info("VibeVoice-ASR: imports OK, will load lazily on first use.")
    except Exception as e:
        logger.warning(f"VibeVoice-ASR unavailable: {e}")

    diarizer = Diarizer()
    profiler = SpeakerProfiler()
    audio_features = AudioFeatureExtractor()
    return whisper, vv_available, diarizer, profiler, audio_features, gpu_engine


def _free_gpu_memory():
    """Release PyTorch CUDA caches."""
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _offload_all_from_gpu() -> None:
    """Move ALL models off GPU before loading VibeVoice (which needs ~9-18 GB)."""
    global _gpu_engine

    # 1. Delete CTranslate2 Whisper model (CT2 manages its own CUDA pool)
    if _whisper is not None:
        _whisper.unload_from_gpu()

    # 2. Move PyTorch models to CPU
    if _profiler is not None:
        try:
            _profiler.to_cpu()
        except Exception:
            pass

    if _audio_features is not None:
        try:
            _audio_features.to_cpu()
        except Exception:
            pass

    if _diarizer is not None:
        try:
            _diarizer.to_cpu()
        except Exception:
            pass

    _free_gpu_memory()
    _gpu_engine = None
    logger.info("All models offloaded from GPU.")


def _reload_all_to_gpu() -> None:
    """Reload support models back to GPU after VibeVoice is done."""
    global _gpu_engine

    if _whisper is not None:
        try:
            _whisper.reload_to_gpu()
        except Exception as e:
            logger.warning(f"Whisper GPU reload failed: {e}")

    if _profiler is not None:
        try:
            _profiler.to_gpu()
        except Exception:
            pass

    if _audio_features is not None:
        try:
            _audio_features.to_gpu()
        except Exception:
            pass

    if _diarizer is not None:
        try:
            _diarizer.to_gpu()
        except Exception:
            pass

    _gpu_engine = "whisper"
    logger.info("Models reloaded to GPU.")


def _unload_vibevoice():
    """Delete VibeVoice model and free all VRAM, then restore other models."""
    global _vibevoice, _gpu_engine
    if _vibevoice is not None:
        logger.info("Unloading VibeVoice-ASR from GPU…")
        try:
            del _vibevoice.model
            del _vibevoice.processor
        except Exception:
            pass
        _vibevoice = None
        _free_gpu_memory()
        _gpu_engine = None
        logger.info("VibeVoice-ASR unloaded.")
    # Reload support models to GPU
    _reload_all_to_gpu()


def _ensure_vibevoice_loaded(progress_cb=None) -> "VibeVoiceTranscriber":
    """Offload all GPU models, then load VibeVoice lazily."""
    global _vibevoice, _gpu_engine

    if _vibevoice is not None:
        return _vibevoice

    if progress_cb:
        progress_cb(3, "Zwalnianie VRAM (offload modeli pomocniczych)…")

    _offload_all_from_gpu()

    if progress_cb:
        progress_cb(5, "Ładowanie VibeVoice-ASR na GPU (bfloat16, ~18 GB, pierwsze uruchomienie ~60 s)…")

    _vibevoice = VibeVoiceTranscriber()
    _gpu_engine = "vibevoice"
    return _vibevoice


def _ensure_whisper_on_gpu():
    """Ensure Whisper is loaded and on GPU (after VV use it may be offloaded)."""
    global _gpu_engine
    if _whisper is not None and not _whisper.is_on_gpu:
        logger.info("Reloading Whisper to GPU…")
        _whisper.reload_to_gpu()
        _gpu_engine = "whisper"


# ─────────────────────────────────────────────────────────────────────────────


def _engine_info(engine: str) -> dict:
    if engine == "vibevoice":
        loaded = _vibevoice is not None
        return {
            "engine": "vibevoice",
            "model": os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR"),
            "loaded": loaded,
            "available": _vibevoice_available,
            "on_gpu": _gpu_engine == "vibevoice",
            "diarization": "built-in",
            "description": "VibeVoice-ASR 9B – single-pass ASR + diarization, 50+ languages (lazy GPU load)",
        }
    loaded = _whisper is not None
    return {
        "engine": "whisper",
        "model": os.getenv("WHISPER_MODEL", "large-v3"),
        "loaded": loaded,
        "available": loaded,
        "on_gpu": _gpu_engine == "whisper",
        "diarization": _diarizer._method if _diarizer else "none",
        "description": f"faster-whisper {os.getenv('WHISPER_MODEL', 'large-v3')} – fast, accurate",
    }


@app.get("/cache/info")
async def cache_info():
    return {
        "transcribe": _transcribe_cache.info(),
        "ollama": _translator_mod._ollama_cache.info(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0-agent",
        "engines": {
            "whisper": _engine_info("whisper"),
            "vibevoice": _engine_info("vibevoice"),
        },
        "profiler": _profiler._method if _profiler else "none",
        "audio_features": _audio_features.loaded_extractors if _audio_features else [],
        "ollama": ollama_available(),
    }


@app.get("/engines")
async def engines_endpoint():
    result = []
    if _whisper is not None:
        result.append(_engine_info("whisper"))
    if _vibevoice_available:
        result.append(_engine_info("vibevoice"))
    return {"engines": result}


@app.get("/models")
async def list_models():
    """Proxy Ollama /api/tags — returns available model names."""
    import httpx as _httpx
    ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    try:
        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            data = resp.json()
        names = [m["name"] for m in (data.get("models") or [])]
        return {"models": names, "default": os.getenv("OLLAMA_MODEL", "gemma4:26b")}
    except Exception as exc:
        return {"models": [], "default": os.getenv("OLLAMA_MODEL", "gemma4:26b"), "error": str(exc)}


@app.delete("/cache")
async def clear_cache_endpoint():
    """Clear all cached transcription results."""
    n = _transcribe_cache.clear()
    return {"cleared": True, "entries_removed": n}


@app.get("/sessions")
async def list_sessions():
    return {"sessions": get_active_sessions()}


@app.post("/session/{session_id}/hint")
async def inject_session_hint(session_id: str, body: dict):
    """Inject a human hint into a running agent session."""
    hint = (body.get("hint") or "").strip()
    if not hint:
        return {"accepted": False, "reason": "empty hint"}
    accepted = inject_hint(session_id, hint)
    return {"accepted": accepted, "session_id": session_id}


@app.post("/session/{session_id}/cancel")
async def cancel_session_endpoint(session_id: str):
    """Request cancellation of a running agent session.

    The agent loop checks the flag between steps and aborts cleanly,
    emitting a `cancelled` SSE event on the original stream.
    """
    accepted = cancel_session(session_id)
    return {"accepted": accepted, "session_id": session_id}


@app.get("/memories")
async def get_memories():
    return {"memories": list_memories()}


@app.delete("/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: str):
    return {"deleted": delete_memory(memory_id)}


@app.delete("/memories")
async def clear_memories_endpoint():
    clear_all_memories()
    return {"cleared": True}


# ── Main transcription endpoint ───────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    translate: bool = Form(default=True),
    engine: str = Form(default="whisper"),
    model: Optional[str] = Form(default=None),  # Ollama model override
):
    """
    Upload audio for agent-driven processing.

    SSE event types streamed:
      agent_start | agent_thinking | tool_call | tool_result | tool_error |
      agent_memory | progress | result | error
    """
    engine = engine.lower().strip()

    # ── Single-analysis policy ────────────────────────────────────────────────
    # Only one analysis may run at a time (single GPU, sequential executor).
    # Reject concurrent uploads with 409 so the UI can surface a clear message
    # instead of silently queueing behind the executor.
    if is_busy():
        active = get_active_sessions()
        return JSONResponse(
            status_code=409,
            content={
                "type": "error",
                "message": "Trwa już inna analiza. Anuluj ją lub poczekaj na zakończenie.",
                "active_sessions": active,
            },
        )

    if engine == "vibevoice" and not _vibevoice_available:
        async def _err():
            yield 'data: {"type":"error","message":"VibeVoice-ASR nie jest dostępny."}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if engine != "vibevoice" and _whisper is None:
        async def _err():
            yield 'data: {"type":"error","message":"Whisper nie jest dostępny."}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    content = await file.read()
    filename = file.filename or "audio"
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    models = {
        "whisper": _whisper,
        "vibevoice_loader": _ensure_vibevoice_loaded,
        "ensure_whisper_gpu": _ensure_whisper_on_gpu,
        "diarizer": _diarizer,
        "profiler": _profiler,
        "audio_features": _audio_features,
        "transcribe_cache": _transcribe_cache,
        "unload_vibevoice": _unload_vibevoice,
    }

    ctx = AgentContext(
        audio_content=content,
        filename=filename,
        language_hint=language or None,
        do_translate=translate,
        queue=queue,
        loop=loop,
        ollama_model=model.strip() if model and model.strip() else None,  # type: ignore[arg-type]
    )

    future = loop.run_in_executor(_executor, lambda: _run_agent_safe(ctx, models))

    async def generate():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=1800.0)
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("result", "error"):
                    break
        except asyncio.TimeoutError:
            yield 'data: {"type":"error","message":"Timeout \u2014 brak aktywno\u015bci przez 30 minut"}\n\n'
        try:
            await future
        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_agent_safe(ctx: AgentContext, models: dict) -> None:
    """Run the agent; always unload vibevoice on exit."""
    try:
        run_agent(ctx, models)
    except Exception as exc:
        logger.exception("Agent top-level error: %s", exc)
        asyncio.run_coroutine_threadsafe(
            ctx.queue.put({"type": "error", "message": str(exc)}),
            ctx.loop,
        )
        asyncio.run_coroutine_threadsafe(ctx.queue.put(None), ctx.loop)
    finally:
        if ctx.asr_engine == "vibevoice":
            _unload_vibevoice()


# ── Fragment re-processing endpoint ───────────────────────────────────────────

@app.post("/reprocess")
async def reprocess(
    file: UploadFile = File(...),
    segments: str = Form(...),          # JSON array of current segments
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    mode: str = Form(...),              # transcription | diarization | translation
    language: Optional[str] = Form(default=None),
    detected_language: Optional[str] = Form(default=None),
    engine: str = Form(default="whisper"),
    model: Optional[str] = Form(default=None),
):
    """Re-process a selected time range of an existing transcript.

    The client sends the source audio + current segments + a [start, end] range
    and a mode. Only the chosen step is re-run on that fragment; the result is
    spliced back and streamed as a fresh `result` event.
    """
    mode = mode.lower().strip()
    if mode not in ("transcription", "diarization", "translation"):
        return JSONResponse(status_code=400,
                            content={"type": "error", "message": f"Nieznany tryb: {mode}"})

    if is_busy():
        return JSONResponse(
            status_code=409,
            content={
                "type": "error",
                "message": "Trwa już inna analiza. Anuluj ją lub poczekaj na zakończenie.",
                "active_sessions": get_active_sessions(),
            },
        )

    if _whisper is None:
        return JSONResponse(status_code=503,
                            content={"type": "error", "message": "Whisper nie jest dostępny."})

    try:
        parsed_segments = json.loads(segments)
        if not isinstance(parsed_segments, list):
            raise ValueError("segments must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(status_code=400,
                            content={"type": "error", "message": f"Nieprawidłowe segmenty: {exc}"})

    content = await file.read()
    filename = file.filename or "audio"
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    models = {
        "whisper": _whisper,
        "vibevoice_loader": _ensure_vibevoice_loaded,
        "ensure_whisper_gpu": _ensure_whisper_on_gpu,
        "diarizer": _diarizer,
        "profiler": _profiler,
        "audio_features": _audio_features,
        "transcribe_cache": _transcribe_cache,
        "unload_vibevoice": _unload_vibevoice,
    }

    ctx = AgentContext(
        audio_content=content,
        filename=filename,
        language_hint=language or None,
        do_translate=False,
        queue=queue,
        loop=loop,
        ollama_model=model.strip() if model and model.strip() else None,  # type: ignore[arg-type]
    )
    # Pre-load existing transcript state into the context.
    ctx.segments = [dict(s) for s in parsed_segments]
    ctx.detected_language = detected_language or (language or "auto")
    ctx.asr_engine = engine.lower().strip()

    future = loop.run_in_executor(
        _executor, lambda: run_reprocess(ctx, models, start_sec, end_sec, mode)
    )

    async def generate():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=1800.0)
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("result", "error"):
                    break
        except asyncio.TimeoutError:
            yield 'data: {"type":"error","message":"Timeout \u2014 brak aktywno\u015bci"}\n\n'
        try:
            await future
        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

