"""Tests for OllamaClient model resolution and fallback logic."""

from __future__ import annotations

import pytest

from pandaro.clients.ollama_client import OllamaClient


class FakeSettings:
    """Minimal settings stub for unit testing."""

    ollama_host = "http://localhost:11434"
    llm_model = "gemma4:31b"
    llm_model_fallback = "gemma3:27b"
    embedding_model = "bge-m3"
    ollama_keep_alive = "0"
    ollama_request_timeout_s = 10.0


class TestModelResolution:
    """Unit-test OllamaClient.resolve_llm_model without a real Ollama."""

    def _client(self, available: list[str]) -> OllamaClient:
        c = OllamaClient(FakeSettings())  # type: ignore[arg-type]

        async def fake_list() -> list[str]:
            return available

        c.list_models = fake_list  # type: ignore[method-assign]
        return c

    @pytest.mark.asyncio
    async def test_exact_match_returned_as_is(self):
        c = self._client(["gemma4:31b", "bge-m3"])
        assert await c.resolve_llm_model() == "gemma4:31b"

    @pytest.mark.asyncio
    async def test_base_name_match_returns_full_tag(self):
        """gemma4 (base) should resolve to gemma4:31b (full tag on Ollama)."""
        c = self._client(["gemma4:31b", "bge-m3"])
        assert await c.resolve_llm_model() == "gemma4:31b"

    @pytest.mark.asyncio
    async def test_fallback_when_primary_absent(self):
        """Falls back to gemma3:27b when gemma4 family is not on Ollama."""
        c = self._client(["gemma3:27b", "bge-m3"])
        assert await c.resolve_llm_model() == "gemma3:27b"

    @pytest.mark.asyncio
    async def test_fallback_base_name_match(self):
        """Fallback base name matches (gemma3:27b → gemma3:latest, etc.)."""
        c = self._client(["gemma3:latest", "bge-m3"])
        # primary gemma4 absent; fallback gemma3:27b base=gemma3 → gemma3:latest
        assert await c.resolve_llm_model() == "gemma3:latest"

    @pytest.mark.asyncio
    async def test_list_models_error_returns_configured_name(self):
        """If Ollama is unreachable, use the configured name as-is."""
        c = OllamaClient(FakeSettings())  # type: ignore[arg-type]

        async def raise_error():
            raise ConnectionError("Ollama unreachable")

        c.list_models = raise_error  # type: ignore[method-assign]
        assert await c.resolve_llm_model() == "gemma4:31b"

    @pytest.mark.asyncio
    async def test_resolution_is_cached(self):
        calls = 0

        async def counting_list() -> list[str]:
            nonlocal calls
            calls += 1
            return ["gemma4:31b"]

        c = OllamaClient(FakeSettings())  # type: ignore[arg-type]
        c.list_models = counting_list  # type: ignore[method-assign]
        await c.resolve_llm_model()
        await c.resolve_llm_model()
        assert calls == 1, "list_models should be called only once (cached)"
