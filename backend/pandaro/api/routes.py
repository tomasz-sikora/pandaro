"""REST + WebSocket routes for sessions, pipeline, search proxy and agent chat."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ..clients import OllamaClient
from ..config import get_settings
from ..gpu import empty_cache, vram_stats
from ..logging_setup import get_logger
from ..orchestrator import Orchestrator
from ..pipeline.export import (
    analysis_to_markdown,
    export_bundle,
    import_bundle,
    transcript_to_srt,
    transcript_to_vtt,
)
from ..schemas import Phase, Preset
from .store import hub

log = get_logger("api")
router = APIRouter(prefix="/api")
orchestrator = Orchestrator()


# --------------------------------------------------------------------------- #
# Health & config                                                             #
# --------------------------------------------------------------------------- #
@router.get("/health")
async def health() -> dict:
    ollama = await OllamaClient().health()
    return {"status": "ok", "ollama": ollama, "vram": vram_stats()}


@router.get("/config")
async def public_config() -> dict:
    s = get_settings()
    return {
        "llm_model": s.llm_model,
        "llm_model_fallback": s.llm_model_fallback,
        "embedding_model": s.embedding_model,
        "default_language": s.default_language,
        "confidence_threshold": s.confidence_threshold,
        "phases": [p.value for p in Phase],
    }


# --------------------------------------------------------------------------- #
# Sessions & pipeline                                                         #
# --------------------------------------------------------------------------- #
@router.post("/sessions")
async def create_session(
    file: UploadFile = File(...),
    preset: str = Form("{}"),
) -> dict:
    s = get_settings()
    data = await file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Plik przekracza limit {s.max_upload_mb} MB")
    try:
        preset_obj = Preset.model_validate_json(preset) if preset.strip() else Preset()
    except Exception as exc:
        raise HTTPException(422, f"Nieprawidłowy preset: {exc}") from exc
    session = hub.create(data, file.filename or "nagranie", preset_obj)
    return {"session_id": session.id, "analysis": session.analysis.model_dump()}


async def _progress_publisher(sid: str):
    async def cb(state) -> None:
        await hub.publish(sid, state)

    return cb


@router.post("/sessions/{sid}/run")
async def run_pipeline(sid: str) -> dict:
    session = hub.get(sid)
    if not session:
        raise HTTPException(404, "Sesja nie istnieje")
    hub._cancelled.discard(sid)  # clear any previous cancel flag
    cb = await _progress_publisher(sid)
    client = OllamaClient()

    async def _job() -> None:
        await orchestrator.run_all(session, cb)
        # After pipeline: free GPU cache and unload Ollama model from VRAM.
        empty_cache()
        await client.unload()
        log.info("pipeline.finished_vram_cleared", sid=sid, **vram_stats())

    asyncio.create_task(_job())
    return {"status": "running"}


@router.post("/sessions/{sid}/cancel")
async def cancel_pipeline(sid: str) -> dict:
    """Ask the running pipeline to stop after the current phase."""
    ok = hub.cancel(sid)
    if not ok:
        raise HTTPException(404, "Sesja nie istnieje")
    return {"status": "cancelled"}


@router.post("/sessions/{sid}/phases/{phase}")
async def rerun_phase(sid: str, phase: str) -> dict:
    session = hub.get(sid)
    if not session:
        raise HTTPException(404, "Sesja nie istnieje")
    try:
        phase_enum = Phase(phase)
    except ValueError as exc:
        raise HTTPException(400, f"Nieznana faza: {phase}") from exc
    hub._cancelled.discard(sid)  # clear cancel so re-run can proceed
    cb = await _progress_publisher(sid)

    async def _job() -> None:
        await orchestrator.run_phase(session, phase_enum, cb)

    asyncio.create_task(_job())
    return {"status": "running", "phase": phase}


@router.get("/sessions/{sid}")
async def get_session(sid: str) -> dict:
    session = hub.get(sid)
    if not session:
        raise HTTPException(404, "Sesja nie istnieje")
    return session.analysis.model_dump()


@router.delete("/sessions/{sid}")
async def clear_session(sid: str) -> dict:
    """The 'Wyczyść' button — server-side ephemerality."""
    ok = hub.clear(sid)
    return {"cleared": ok}


@router.websocket("/sessions/{sid}/ws")
async def session_ws(websocket: WebSocket, sid: str) -> None:
    await websocket.accept()
    q = hub.subscribe(sid)
    try:
        while True:
            state = await q.get()
            await websocket.send_json(state.model_dump())
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(sid, q)


# --------------------------------------------------------------------------- #
# Export / import                                                             #
# --------------------------------------------------------------------------- #
@router.get("/sessions/{sid}/export")
async def export_session(sid: str, fmt: str = "pandaro"):
    session = hub.get(sid)
    if not session:
        raise HTTPException(404, "Sesja nie istnieje")
    a = session.analysis
    if fmt == "pandaro":
        return JSONResponse(json.loads(export_bundle(a)))
    if fmt == "srt":
        return PlainTextResponse(transcript_to_srt(a.transcript), media_type="text/plain")
    if fmt == "vtt":
        return PlainTextResponse(transcript_to_vtt(a.transcript), media_type="text/vtt")
    if fmt == "md":
        return PlainTextResponse(analysis_to_markdown(a), media_type="text/markdown")
    raise HTTPException(400, f"Nieznany format: {fmt}")


@router.post("/import")
async def import_session(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        analysis = import_bundle(data)
    except Exception as exc:
        raise HTTPException(422, f"Nieprawidłowy plik .pandaro: {exc}") from exc
    session = hub.create_from_analysis(analysis)
    return {"session_id": session.id, "analysis": session.analysis.model_dump()}


# --------------------------------------------------------------------------- #
# Embeddings proxy (browser RAG query) + agent chat                           #
# --------------------------------------------------------------------------- #
class EmbedRequest(BaseModel):
    texts: list[str]


@router.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    vectors = await OllamaClient().embed(req.texts)
    return {"embeddings": vectors}


class ChatRequest(BaseModel):
    messages: list[dict]
    context: list[dict] = []  # retrieved RAG chunks [{text, speaker, start, end}]
    temperature: float = 0.2


def _build_chat_system(context_chunks: list[dict]) -> str:
    if context_chunks:
        context_block = "\n\n".join(
            "[{n}] [{spk} @ {t:.0f}s] {txt}".format(
                n=i + 1, spk=c.get("speaker", "?"),
                t=float(c.get("start", 0)), txt=c.get("text", "")
            )
            for i, c in enumerate(context_chunks)
        )
        return (
            "Jesteś asystentem analizującym nagranie rozmowy. Odpowiadaj po polsku, "
            "wyłącznie na podstawie dostarczonych fragmentów transkryptu. "
            "Cytuj fragmenty w formacie [nr] [ROZMÓWCA, mm:ss] jako dowód. "
            "Jeśli brak informacji w transkrypcie, powiedz to wprost.\n\n"
            f"FRAGMENTY TRANSKRYPTU:\n{context_block}"
        )
    return (
        "Jesteś asystentem analizującym nagranie rozmowy. "
        "Odpowiadaj po polsku na podstawie historii konwersacji."
    )


@router.post("/llm/chat")
async def chat(req: ChatRequest) -> dict:
    """RAG-grounded blocking chat (legacy; prefer /llm/chat/stream)."""
    client = OllamaClient()
    system = _build_chat_system(req.context)
    messages = [{"role": "system", "content": system}, *req.messages]
    answer = await client.chat(messages, temperature=req.temperature)
    return {"answer": answer}


@router.post("/llm/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming SSE chat. Yields ``data: {"chunk": "..."}`` lines.
    Special events: ``{"thinking": true}`` when model is reasoning,
    ``{"thinking": false}`` when reasoning ends.
    Terminated by ``data: [DONE]``."""
    client = OllamaClient()
    system = _build_chat_system(req.context)
    messages = [{"role": "system", "content": system}, *req.messages]

    async def event_stream():
        try:
            async for chunk in client.stream_chat(messages, temperature=req.temperature):
                if chunk == "\x00THINKING\x00":
                    yield f"data: {json.dumps({'thinking': True})}\n\n"
                elif chunk == "\x00DONE_THINKING\x00":
                    yield f"data: {json.dumps({'thinking': False})}\n\n"
                else:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as exc:
            log.warning("chat_stream.error", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
