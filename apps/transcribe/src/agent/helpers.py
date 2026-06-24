from __future__ import annotations
from .context import AgentContext
import asyncio, io, json, logging, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx, numpy as np
logger = logging.getLogger(__name__)

# ── Re-export constants used by other modules ─────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHECKPOINT_DIR = Path(os.getenv("AGENT_CHECKPOINT_DIR", "/app/data/checkpoints"))
ARTIFACTS_DIR = Path(os.getenv("AGENT_ARTIFACTS_DIR", "/app/data/artifacts"))
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "50"))
MAX_RETRIES_PER_TOOL = 3
# ── Single unified context window ─────────────────────────────────────────────
# Using ONE num_ctx for EVERY Ollama call prevents model reloads between turns.
# Changing num_ctx mid-session forces Ollama to unload + reload the model and
# flush the KV cache — the primary cause of multi-minute delays.
# 32768 tokens is big enough for summaries (gemma4:26b handles it fine) and
# fast enough for tool-calling decisions (<2s per turn on a single GPU).
_UNIFIED_CTX     = int(os.getenv("AGENT_NUM_CTX", "32768"))
AGENT_NUM_CTX    = _UNIFIED_CTX   # kept for external imports
SUMMARY_NUM_CTX  = _UNIFIED_CTX   # kept for external imports
AGENT_TEMPERATURE = 0.05
KEEP_ALIVE = -1  # keep Ollama model resident in VRAM indefinitely

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


def _format_transcript_context(
    ctx: "AgentContext",
    window_size: Optional[int] = None,
    speaker_only: Optional[str] = None,
    include_tags: bool = True,
) -> str:
    """
    Return a windowed, human-readable transcript for inclusion in LLM prompts.

    window_size: max number of segments to include (None → ctx.max_ctx_segments).
    speaker_only: filter to a single speaker label.
    include_tags: append [tag] annotations for tagged segments.

    Always includes the last `window_size` segments so context stays current.
    For very long transcripts the agent never receives the full text in one shot.
    """
    segs = ctx.segments
    if speaker_only:
        segs = [s for s in segs if s.get("speaker") == speaker_only]
    n = window_size or ctx.max_ctx_segments
    if len(segs) > n:
        segs = segs[-n:]
    lines = []
    for s in segs:
        sp = ctx.display_names.get(s.get("speaker", ""), s.get("speaker", "?"))
        text = (s.get("text_pl") or s.get("text") or "").strip()
        t = int(s.get("start") or 0)
        m, sec = divmod(t, 60)
        ts = f"{m}:{sec:02d}"
        tag_str = ""
        if include_tags and s.get("id") in ctx.segment_tags:
            tags = ctx.segment_tags[s["id"]]
            tag_str = " [" + ", ".join(tags) + "]"
        lines.append(f"[{ts} {sp}]{tag_str} {text}")
    return "\n".join(lines)


def _call_ollama_chat(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    num_ctx: int = AGENT_NUM_CTX,
    timeout: float = 300.0,
    model: Optional[str] = None,
) -> Optional[Dict]:
    body: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": _UNIFIED_CTX,   # always use unified value — never override
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
    timeout: float = 600.0,
    json_format: Optional[Dict] = None,
    model: Optional[str] = None,
    num_predict: int = 0,   # 0 = model default; set >0 to cap output length
) -> str:
    body: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        # cache_prompt=True: Ollama persists the evaluated prompt prefix in the
        # KV cache so follow-up calls with the same prefix skip re-prefill.
        "options": {
            "num_ctx": _UNIFIED_CTX,   # must match chat to avoid model reload
            "temperature": AGENT_TEMPERATURE,
            **({"num_predict": num_predict} if num_predict > 0 else {}),
        },
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

