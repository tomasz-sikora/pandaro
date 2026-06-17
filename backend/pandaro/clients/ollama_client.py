"""Async Ollama client: LLM generation (streaming + blocking) and embeddings.

The browser cannot reach the GPU, so all LLM/embedding traffic is proxied
through the backend to Ollama. We pass ``keep_alive=0`` so models are
unloaded from VRAM immediately after each call, freeing memory for ASR/etc.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ..config import Settings, get_settings
from ..logging_setup import get_logger

log = get_logger("ollama")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._resolved_llm: str | None = None

    # --- introspection --------------------------------------------------------
    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.s.ollama_host}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]

    async def health(self) -> dict:
        try:
            models = await self.list_models()
            return {"ok": True, "models": models, "llm": await self.resolve_llm_model()}
        except Exception as exc:  # pragma: no cover
            log.warning("ollama.health_failed", error=str(exc))
            return {"ok": False, "error": "Ollama jest nieosiągalna."}

    async def resolve_llm_model(self) -> str:
        """Return the actual Ollama tag for the configured LLM model.

        Matches by exact tag first, then by base name (e.g. ``gemma4:31b``
        satisfies config ``gemma4``). Falls back to ``llm_model_fallback``
        if neither matches.
        """
        if self._resolved_llm:
            return self._resolved_llm
        wanted = self.s.llm_model
        try:
            available = await self.list_models()
        except Exception:
            self._resolved_llm = wanted
            return wanted

        def find(tag: str) -> str | None:
            if tag in available:
                return tag
            base = tag.split(":")[0]
            for m in available:
                if m.split(":")[0] == base:
                    return m
            return None

        resolved = find(wanted) or find(self.s.llm_model_fallback) or wanted
        self._resolved_llm = resolved
        if resolved != wanted:
            log.warning("ollama.llm_fallback", wanted=wanted, using=resolved)
        return self._resolved_llm

    # --- blocking generation --------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        model = model or await self.resolve_llm_model()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.s.ollama_request_timeout_s) as client:
            r = await client.post(f"{self.s.ollama_host}/api/generate", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"generate failed: {r.status_code} {r.text}")
            return r.json().get("response", "").strip()

    # --- blocking chat --------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        model = model or await self.resolve_llm_model()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.s.ollama_request_timeout_s) as client:
            r = await client.post(f"{self.s.ollama_host}/api/chat", json=payload)
            if r.status_code != 200:
                raise OllamaError(f"chat failed: {r.status_code} {r.text}")
            return r.json().get("message", {}).get("content", "").strip()

    # --- streaming chat -------------------------------------------------------
    async def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        model = model or await self.resolve_llm_model()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.s.ollama_request_timeout_s) as client:
            async with client.stream(
                "POST", f"{self.s.ollama_host}/api/chat", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise OllamaError(f"stream_chat failed: {resp.status_code} {body.decode()}")
                thinking = False
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    # Gemma-4 and other thinking models emit reasoning in "thinking"
                    # before the actual response. Yield a special marker on first
                    # thinking token so the frontend can show a progress indicator.
                    if msg.get("thinking"):
                        if not thinking:
                            thinking = True
                            yield "\x00THINKING\x00"  # sentinel for UI
                        continue
                    chunk = msg.get("content", "")
                    if chunk:
                        if thinking:
                            thinking = False
                            yield "\x00DONE_THINKING\x00"  # thinking finished
                        yield chunk

    # --- embeddings -----------------------------------------------------------
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self.s.embedding_model
        async with httpx.AsyncClient(timeout=self.s.ollama_request_timeout_s) as client:
            r = await client.post(
                f"{self.s.ollama_host}/api/embed",
                json={"model": model, "input": texts, "keep_alive": self.s.ollama_keep_alive},
            )
            if r.status_code != 200:
                raise OllamaError(f"embed failed: {r.status_code} {r.text}")
            return r.json().get("embeddings") or []

    # --- VRAM cleanup ---------------------------------------------------------
    async def unload(self) -> None:
        """Force Ollama to unload the resident LLM from VRAM (keep_alive=0)."""
        model = self._resolved_llm or self.s.llm_model
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self.s.ollama_host}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "0"},
                )
            log.info("ollama.unloaded", model=model)
        except Exception as exc:
            log.warning("ollama.unload_failed", model=model, error=str(exc))
