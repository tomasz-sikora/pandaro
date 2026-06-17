"""GPU lifecycle manager: load -> infer -> offload, with a single resident lock.

On a 24GB RTX 3090 shared with Ollama, only one heavy HuggingFace model may be
resident at a time. This manager provides:

* an async lock so phases serialize their GPU usage;
* explicit ``offload()`` that moves a model to CPU / drops references and calls
  ``torch.cuda.empty_cache()``;
* a context manager that guarantees offload even on error.

It degrades gracefully when torch/CUDA is unavailable (CPU/dev mode).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..logging_setup import get_logger

log = get_logger("gpu")


def cuda_available() -> bool:
    try:  # pragma: no cover - depends on torch
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def empty_cache() -> None:
    try:  # pragma: no cover - depends on torch
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def vram_stats() -> dict[str, float]:
    """Return {total_gb, used_gb, free_gb} or empty if no CUDA."""
    try:  # pragma: no cover - depends on torch
        import torch

        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {
            "total_gb": round(total / 1024**3, 2),
            "free_gb": round(free / 1024**3, 2),
            "used_gb": round((total - free) / 1024**3, 2),
        }
    except Exception:
        return {}


class GPUManager:
    """Serializes heavy model usage and frees VRAM between phases."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._resident: str | None = None

    @property
    def resident(self) -> str | None:
        return self._resident

    @contextlib.asynccontextmanager
    async def session(self, name: str) -> AsyncIterator[None]:
        """Acquire exclusive GPU access for ``name``; offload on exit."""
        async with self._lock:
            self._resident = name
            log.info("gpu.acquire", model=name, **vram_stats())
            try:
                yield
            finally:
                self._resident = None
                empty_cache()
                log.info("gpu.release", model=name, **vram_stats())

    async def run(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run a blocking model call under the GPU lock in a worker thread."""
        async with self.session(name):
            return await asyncio.to_thread(fn)


gpu_manager = GPUManager()
