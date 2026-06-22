"""
FastAPI transcription service with SSE streaming progress.

POST /transcribe
  multipart/form-data: file, language?, translate?, engine?
  Returns text/event-stream → progress events + final result.

GET /health  → engines status, diarizer, profiler, ollama
GET /engines → list of available (loaded) engines

VRAM strategy:
  - Whisper is loaded at startup (GPU, ~6 GB).
  - VibeVoice (9B, ~18 GB) is loaded LAZILY on first use and unloaded
    afterwards to free VRAM.  Only one ASR engine occupies the GPU at a time.
"""
import asyncio
import io
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydub import AudioSegment

from .transcriber import WhisperTranscriber
from .vibevoice_transcriber import VibeVoiceTranscriber
from .nemotron_transcriber import NemotronTranscriber
from .diarizer import Diarizer
from .speaker_profiler import SpeakerProfiler
from .audio_features import AudioFeatureExtractor
from .translator import translate_segments_to_polish, ollama_available
from .speaker_identifier import identify_speakers
from .cache import LRUCache
from . import translator as _translator_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Help PyTorch avoid VRAM fragmentation when models are swapped
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = FastAPI(title="Heimdall Transcription Service", version="2.0.0")

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

# Whisper loaded at startup; VibeVoice and Nemotron loaded lazily per first use
_whisper: Optional[WhisperTranscriber] = None
_vibevoice: Optional[VibeVoiceTranscriber] = None   # None = not yet loaded; loaded on first VV request
_vibevoice_available: bool = False                   # True = imports OK, can be loaded
_nemotron: Optional[NemotronTranscriber] = None      # None = not yet loaded; loaded lazily
_nemotron_available: bool = False                    # True = NeMo imports OK
_diarizer: Optional[Diarizer] = None
_profiler: Optional[SpeakerProfiler] = None
_audio_features: Optional[AudioFeatureExtractor] = None
# Track which engine currently holds GPU memory
_gpu_engine: Optional[str] = None  # "whisper" | "vibevoice" | "nemotron" | None


@app.on_event("startup")
async def startup():
    global _whisper, _vibevoice_available, _nemotron_available, _diarizer, _profiler, _audio_features, _gpu_engine
    loop = asyncio.get_event_loop()
    logger.info("Loading Whisper at startup (VibeVoice and Nemotron load lazily on first use)…")
    result = await loop.run_in_executor(_executor, _load_startup_models)
    _whisper, _vibevoice_available, _nemotron_available, _diarizer, _profiler, _audio_features, _gpu_engine = result
    logger.info("Startup models loaded.")


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

    # Check Nemotron importability without loading weights
    nemotron_available = False
    try:
        import nemo.collections.asr  # noqa: F401
        nemotron_available = True
        logger.info("Nemotron 3.5 ASR: NeMo imports OK, will load lazily on first use.")
    except Exception as e:
        logger.warning(f"Nemotron 3.5 ASR unavailable (NeMo not installed?): {e}")

    diarizer = Diarizer()
    profiler = SpeakerProfiler()
    audio_features = AudioFeatureExtractor()
    return whisper, vv_available, nemotron_available, diarizer, profiler, audio_features, gpu_engine


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

    # 5. Offload Nemotron if loaded (small model, but free VRAM for VibeVoice)
    if _nemotron is not None:
        try:
            _nemotron.unload_from_gpu()
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

    if _nemotron is not None:
        try:
            _nemotron.reload_to_gpu()
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


def _ensure_nemotron_loaded(progress_cb=None) -> "NemotronTranscriber":
    """Load Nemotron lazily on first use (stays loaded — small model ~1.2 GB)."""
    global _nemotron, _gpu_engine

    if _nemotron is not None:
        if not _nemotron.is_on_gpu:
            _nemotron.reload_to_gpu()
        return _nemotron

    if progress_cb:
        progress_cb(5, "Ładowanie Nemotron 3.5 ASR na GPU (~1.2 GB, pierwsze uruchomienie ~30 s)…")

    _nemotron = NemotronTranscriber()
    _gpu_engine = "nemotron"
    return _nemotron


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
    if engine == "nemotron":
        loaded = _nemotron is not None
        return {
            "engine": "nemotron",
            "model": os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
            "loaded": loaded,
            "available": _nemotron_available,
            "on_gpu": _nemotron.is_on_gpu if loaded else False,
            "diarization": "pyannote",
            "description": "Nemotron 3.5 ASR 600 M – cache-aware FastConformer-RNNT, 40 language-locales (lazy GPU load, ~1.2 GB)",
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
    """Return current LRU cache statistics."""
    return {
        "transcribe": _transcribe_cache.info(),
        "ollama": _translator_mod._ollama_cache.info(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_source": "local (HuggingFace weights)",
        "engines": {
            "whisper": _engine_info("whisper"),
            "vibevoice": _engine_info("vibevoice"),
            "nemotron": _engine_info("nemotron"),
        },
        "profiler": _profiler._method if _profiler else "none",
        "audio_features": _audio_features.loaded_extractors if _audio_features else [],
        "ollama": ollama_available(),
    }


@app.get("/engines")
async def engines_endpoint():
    """List available engines (Whisper always, VibeVoice if imports OK)."""
    result = []
    if _whisper is not None:
        result.append(_engine_info("whisper"))
    if _vibevoice_available:
        result.append(_engine_info("vibevoice"))
    if _nemotron_available:
        result.append(_engine_info("nemotron"))
    return {"engines": result}


# ─────────────────────────────────────────────────────────────────────────────


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    translate: bool = Form(default=True),
    engine: str = Form(default="whisper"),
):
    engine = engine.lower().strip()

    # Validate availability (not whether currently loaded on GPU)
    if engine == "vibevoice":
        if not _vibevoice_available:
            async def _err():
                yield f'data: {{"type":"error","message":"VibeVoice-ASR nie jest dostępny (błąd importu)."}}\n\n'
            return StreamingResponse(
                _err(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    elif engine == "nemotron":
        if not _nemotron_available:
            async def _err():
                yield f'data: {{"type":"error","message":"Nemotron 3.5 ASR nie jest dostępny (NeMo nie zainstalowane)."}}\n\n'
            return StreamingResponse(
                _err(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    else:
        engine = "whisper"
        if _whisper is None:
            async def _err():
                yield f'data: {{"type":"error","message":"Whisper nie jest dostępny (błąd ładowania)."}}\n\n'
            return StreamingResponse(
                _err(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    content = await file.read()
    filename = file.filename or "audio"
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    future = loop.run_in_executor(
        _executor,
        lambda: _pipeline(content, filename, language, translate, engine, queue, loop),
    )

    async def generate():
        try:
            while True:
                # 30-minute inter-event timeout to handle very long (2h+) recordings.
                # The pipeline emits a progress event at least every ~15 segments, so
                # 30 minutes is generous even for the slowest model/hardware combo.
                event = await asyncio.wait_for(queue.get(), timeout=1800.0)
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("result", "error"):
                    break
        except asyncio.TimeoutError:
            yield f'data: {{"type":"error","message":"Timeout — brak aktywności przez 30 minut"}}\n\n'
        try:
            await future
        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────


def _send(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, event: dict):
    asyncio.run_coroutine_threadsafe(queue.put(event), loop)


def _progress(queue, loop, stage: str, pct: int, msg: str):
    _send(queue, loop, {"type": "progress", "stage": stage, "progress": pct, "message": msg})


def _pipeline(
    content: bytes,
    filename: str,
    language: Optional[str],
    do_translate: bool,
    engine: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
):
    import hashlib
    is_vibevoice = (engine == "vibevoice")
    is_nemotron = (engine == "nemotron")

    # ── Cache check ─────────────────────────────────────────────────────
    audio_sha = hashlib.sha256(content).hexdigest()
    cache_key = _transcribe_cache.key(audio_sha, language, engine, do_translate)
    cached_result = _transcribe_cache.get(cache_key)
    if cached_result is not None:
        _progress(queue, loop, "done", 100, "Wynik z cache (bez ponownego przetwarzania).")
        _send(queue, loop, {**cached_result, "cached": True})
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        return

    # Resolve / lazy-load the transcriber inside the executor thread
    def _prog(pct, msg):
        _progress(queue, loop, "loading_model" if pct < 15 else "transcribing", pct, msg)

    if is_vibevoice:
        try:
            transcriber = _ensure_vibevoice_loaded(progress_cb=_prog)
        except Exception as exc:
            logger.exception("VibeVoice load failed during request")
            _send(queue, loop, {"type": "error", "message": f"Błąd ładowania VibeVoice: {exc}"})
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            return
    elif is_nemotron:
        try:
            transcriber = _ensure_nemotron_loaded(progress_cb=_prog)
        except Exception as exc:
            logger.exception("Nemotron load failed during request")
            _send(queue, loop, {"type": "error", "message": f"Błąd ładowania Nemotron: {exc}"})
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            return
    else:
        transcriber = _whisper
        # Ensure Whisper is the active GPU engine
        _ensure_whisper_on_gpu()

    try:
        # ── 1. Decode audio ─────────────────────────────────────────────────
        _progress(queue, loop, "decoding", 5, "Dekodowanie audio…")
        audio, sr = _decode_audio(content, filename)
        duration = len(audio) / sr
        _progress(queue, loop, "decoding", 10, f"Czas trwania: {duration:.1f}s")

        # ── 2. Transcribe ────────────────────────────────────────────────────
        engine_label = "VibeVoice-ASR" if is_vibevoice else f"Whisper {os.getenv('WHISPER_MODEL','large-v3')}"
        _progress(queue, loop, "loading_model", 12, f"Silnik: {engine_label}…")

        chunks, detected_lang, dur = transcriber.transcribe(
            audio, language,
            progress_cb=lambda pct, msg: _progress(queue, loop, "transcribing", pct, msg),
        )

        if not chunks:
            raise ValueError("Transkrypcja nie zwróciła żadnych segmentów.")

        _progress(queue, loop, "transcribing", 62, f"Wykryty język: {detected_lang}")

        # ── 3. Diarize (VibeVoice has built-in diarization; others use pyannote) ───
        if is_vibevoice:
            _progress(queue, loop, "diarizing", 72, "Diaryzacja wbudowana (VibeVoice).")
        else:
            _progress(queue, loop, "diarizing", 65, "Identyfikacja mówców…")
            chunks = _diarizer.diarize(audio, sr, chunks)
            _progress(queue, loop, "diarizing", 72, "Zidentyfikowano mówców.")

        # ── 4. Speaker profiling ─────────────────────────────────────────────
        _progress(queue, loop, "profiling", 74, "Analiza cech mówców (płeć, wiek)…")
        speaker_profiles_raw = _profiler.profile_speakers(audio, sr, chunks)
        _progress(queue, loop, "profiling", 78, "Profil głosu gotowy.")

        # ── 5. Audio feature extraction (emotion, speech rate, SNR) ──────────
        audio_features_raw: dict = {}
        if _audio_features and _audio_features.loaded_extractors:
            _progress(queue, loop, "profiling", 79, f"Ekstrakcja cech audio ({', '.join(_audio_features.loaded_extractors)})…")
            try:
                audio_features_raw = _audio_features.extract_per_speaker(audio, sr, chunks)
            except Exception as exc:
                logger.warning(f"Audio feature extraction failed: {exc}")
        _progress(queue, loop, "profiling", 80, "Analiza gotowa.")

        # ── 5. Translate to Polish ───────────────────────────────────────────
        if do_translate and detected_lang != "pl":
            _progress(queue, loop, "translating", 82, "Tłumaczenie na polski (via Ollama)…")
            chunks = translate_segments_to_polish(chunks, detected_lang)
            _progress(queue, loop, "translating", 92, "Tłumaczenie gotowe.")
        else:
            for c in chunks:
                c["text_pl"] = c["text"]

        # ── 6. Identify speakers (name extraction via LLM + gender fallback) ──
        _progress(queue, loop, "identifying", 94, "Identyfikacja mówców (LLM)…")
        try:
            display_names = identify_speakers(chunks, speaker_profiles_raw)
            logger.info(f"Speaker names identified: {display_names}")
        except Exception as exc:
            logger.warning(f"Speaker identification failed: {exc}")
            display_names = {}

        # ── 7. Build result ──────────────────────────────────────────────────
        segments = [
            {
                "id": i,
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
                "text_pl": c.get("text_pl") or c["text"],
                "speaker": c["speaker"],
                "language": detected_lang,
                "words": c.get("words") or [],
            }
            for i, c in enumerate(chunks)
        ]

        speaker_profiles = {
            sp: {
                "gender": p.get("gender"),
                "gender_probs": p.get("gender_probs"),
                "age_estimate": p.get("age_estimate"),
                "age_group": p.get("age_group"),
                "confidence": p.get("confidence"),
                "display_name": display_names.get(sp),
                # merge audio features (emotion, speech_rate, snr, …)
                **audio_features_raw.get(sp, {}),
            }
            for sp, p in speaker_profiles_raw.items()
        }

        _progress(queue, loop, "done", 100, "Gotowe!")
        result_event = {
            "type": "result",
            "segments": segments,
            "detected_language": detected_lang,
            "duration": dur,
            "speaker_profiles": speaker_profiles,
            "model_used": (
                os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR")
                if is_vibevoice
                else os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
                if is_nemotron
                else os.getenv("WHISPER_MODEL", "large-v3")
            ),
            "asr_engine": engine,
        }
        _transcribe_cache.put(cache_key, result_event)
        _send(queue, loop, result_event)

    except Exception as exc:
        logger.exception("Pipeline error")
        _send(queue, loop, {"type": "error", "message": str(exc)})
    finally:
        # Always unload VibeVoice after use to free VRAM for other models/tasks.
        # Nemotron stays loaded (small model, ~1.2 GB) — no explicit unload needed.
        if is_vibevoice:
            _unload_vibevoice()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


# ─────────────────────────────────────────────────────────────────────────────


def _decode_audio(content: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Convert any audio format to 16 kHz mono float32."""
    TARGET_SR = 16_000
    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(content))
    except Exception:
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            audio_seg = AudioSegment.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    audio_seg = audio_seg.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2 ** (audio_seg.sample_width * 8 - 1))
    return samples, TARGET_SR
