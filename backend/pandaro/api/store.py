from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
import uuid
from pathlib import Path

from ..logging_setup import get_logger
from ..orchestrator import Session
from ..schemas import Analysis, PhaseState, Preset

log = get_logger("store")


class SessionHub:
    """Holds sessions and broadcasts phase-progress events to WS subscribers."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._workdirs: dict[str, str] = {}
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._cancelled: set[str] = set()

    # --- lifecycle --------------------------------------------------------
    def create(self, audio_bytes: bytes, filename: str, preset: Preset) -> Session:
        sid = uuid.uuid4().hex
        workdir = tempfile.mkdtemp(prefix=f"pandaro_{sid}_")
        self._workdirs[sid] = workdir
        suffix = Path(filename).suffix or ".bin"
        audio_path = str(Path(workdir) / f"source{suffix}")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        session = Session(sid, audio_path, preset, filename)
        self._sessions[sid] = session
        self._queues[sid] = []
        self._cancelled.discard(sid)
        log.info("session.create", sid=sid, filename=filename, bytes=len(audio_bytes))
        return session

    def create_from_analysis(self, analysis: Analysis) -> Session:
        """Reload an imported analysis (no audio) into a fresh session."""
        sid = uuid.uuid4().hex
        session = Session(sid, "", analysis.preset, analysis.media_filename)
        session.analysis = analysis
        self._sessions[sid] = session
        self._queues[sid] = []
        self._cancelled.discard(sid)
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def cancel(self, sid: str) -> bool:
        """Signal that the running pipeline for *sid* should stop after the
        current phase. Returns True if the session exists."""
        if sid not in self._sessions:
            return False
        self._cancelled.add(sid)
        log.info("session.cancel", sid=sid)
        return True

    def is_cancelled(self, sid: str) -> bool:
        return sid in self._cancelled

    def clear(self, sid: str) -> bool:
        """Wipe a session and its temp files (the 'Wyczyść' button)."""
        session = self._sessions.pop(sid, None)
        self._queues.pop(sid, None)
        self._cancelled.discard(sid)
        workdir = self._workdirs.pop(sid, None)
        if workdir:
            with contextlib.suppress(Exception):
                shutil.rmtree(workdir, ignore_errors=True)
        if session:
            log.info("session.clear", sid=sid)
        return session is not None

    def clear_all(self) -> None:
        for sid in list(self._sessions):
            self.clear(sid)

    # --- event hub --------------------------------------------------------
    def subscribe(self, sid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(sid, []).append(q)
        return q

    def unsubscribe(self, sid: str, q: asyncio.Queue) -> None:
        if sid in self._queues and q in self._queues[sid]:
            self._queues[sid].remove(q)

    async def publish(self, sid: str, state: PhaseState) -> None:
        for q in list(self._queues.get(sid, [])):
            await q.put(state)


hub = SessionHub()
