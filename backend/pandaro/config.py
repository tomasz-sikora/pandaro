"""Application configuration (12-factor, env-driven).

All knobs that affect deployment, hardware budget or model selection live here so
they can be overridden via environment variables (see ``config/.env.example``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PANDARO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 9090
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:9090"])
    # Directory containing the built SPA (index.html + assets). Optional in dev.
    frontend_dist: str = "frontend/dist"
    # Max upload size in megabytes (2h recordings can be large).
    max_upload_mb: int = 2048

    # --- Ollama (LLM + embeddings live on the host) -----------------------
    ollama_host: str = "http://localhost:11434"
    # User asked for "Gemma 4"; falls back to gemma3:27b if that tag is absent.
    llm_model: str = "gemma4"
    llm_model_fallback: str = "gemma3:27b"
    embedding_model: str = "bge-m3"
    # keep_alive passed to Ollama; "0" unloads immediately to free VRAM.
    ollama_keep_alive: str = "0"
    ollama_request_timeout_s: float = 600.0

    # --- HuggingFace ------------------------------------------------------
    hf_token: str | None = None
    hf_home: str | None = None  # cache dir, mounted as a volume in Docker

    # --- ASR --------------------------------------------------------------
    asr_backend: str = "faster-whisper"  # faster-whisper | whisperx | stub
    asr_model: str = "large-v3"
    asr_compute_type: str = "float16"  # float16 | int8_float16 | int8
    asr_beam_size: int = 5
    # Below this aggregated per-word probability, a word is "low confidence".
    confidence_threshold: float = 0.55

    # --- Diarization ------------------------------------------------------
    diarization_backend: str = "pyannote"  # pyannote | nemo | stub
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # --- Paralinguistics --------------------------------------------------
    age_gender_model: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    emotion_model: str = "audeering/wav2vec2-large-robust-24-ft-emotion-msp-dim"

    # --- Hardware / VRAM --------------------------------------------------
    device: str = "auto"  # auto | cuda | cpu
    # Only one heavy model resident at a time on a 24GB card shared with Ollama.
    max_resident_models: int = 1
    low_vram: bool = False

    # --- Pipeline defaults ------------------------------------------------
    default_language: str = "pl"
    translate_target: str = "pl"
    # Map-reduce summarization chunk budget (in characters, ~ tokens*4).
    summary_chunk_chars: int = 8000

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:  # pragma: no cover - depends on torch presence
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
