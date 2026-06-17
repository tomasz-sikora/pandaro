"""Async Ollama client for LLM generation and embeddings (runs on the host).

The browser cannot reach the GPU, so all LLM/embedding traffic is proxied
through the backend to Ollama. We pass ``keep_alive`` so Gemma is unloaded from
VRAM when the heavy ASR/diarization phases need the card.
"""

from __future__ import annotations

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

    # --- introspection ----------------------------------------------------
    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.s.ollama_host}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]

    async def health(self) -> dict:
        try:
            models = await self.list_models()
            return {"ok": True, "models": models, "llm": await self.resolve_llm_model()}
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("ollama.health_failed", error=str(exc))
            return {"ok": False, "error": "Ollama jest nieosiągalna."}

    async def resolve_llm_model(self) -> str:
        """Use the configured LLM model, falling back if its tag is absent.

        Honours the user's request for "gemma4" but degrades to gemma3:27b when
        the newer tag is not pulled on the host Ollama.
        """
        if self._resolved_llm:
            return self._resolved_llm
        wanted = self.s.llm_model
        try:
            available = await self.list_models()
        except Exception:
            self._resolved_llm = wanted
            return wanted

        def has(tag: str) -> bool:
            base = tag.split(":")[0]
            return any(m == tag or m.split(":")[0] == base for m in available)

        self._resolved_llm = wanted if has(wanted) else self.s.llm_model_fallback
        if self._resolved_llm != wanted:
            log.warning("ollama.llm_fallback", wanted=wanted, using=self._resolved_llm)
        return self._resolved_llm

    # --- generation -------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        model = model or await self.resolve_llm_model()
        payload = {
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

    # --- embeddings -------------------------------------------------------
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self.s.embedding_model
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.s.ollama_request_timeout_s) as client:
            # Ollama's /api/embed accepts a batch via "input".
            r = await client.post(
                f"{self.s.ollama_host}/api/embed",
                json={"model": model, "input": texts, "keep_alive": self.s.ollama_keep_alive},
            )
            if r.status_code != 200:
                raise OllamaError(f"embed failed: {r.status_code} {r.text}")
            data = r.json()
            out = data.get("embeddings") or []
        return out
