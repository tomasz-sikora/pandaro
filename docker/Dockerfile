# syntax=docker/dockerfile:1.7

# ----------------------------------------------------------------------------
# Pandaro — multi-stage image
#   stage 1 (web):  build the Polish SPA with Vite → /frontend/dist
#   stage 2 (app):  CUDA runtime + Python backend, serves the built SPA on :8080
# ----------------------------------------------------------------------------

# ---- stage 1: frontend build ----------------------------------------------
FROM node:22-bookworm-slim AS web
WORKDIR /frontend

# Install deps first (better layer caching). BuildKit cache mount keeps the npm
# cache across builds so packages are not re-downloaded.
COPY frontend/package.json frontend/package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build   # → /frontend/dist


# ---- stage 2: backend + GPU runtime ---------------------------------------
# CUDA 12.1 runtime matches the torch cu121 wheels installed below.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All HF caches live under one volume-mounted dir to avoid re-downloads.
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/torch \
    PANDARO_FRONTEND_DIST=/app/frontend/dist \
    PANDARO_HF_HOME=/cache/huggingface

# System deps: python, ffmpeg (audio decode), libsndfile (soundfile), git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# 1) Torch first (CUDA 12.1 wheels), cached. Pin to keep ctranslate2/whisperx happy.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchaudio

# 2) Core + ML Python deps.
COPY backend/requirements.txt backend/requirements-ml.txt ./backend/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r backend/requirements.txt -r backend/requirements-ml.txt \
        huggingface_hub

# 3) Backend source.
COPY backend/ ./backend/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./backend

# 4) Built SPA from stage 1.
COPY --from=web /frontend/dist ./frontend/dist

# 5) Preload HF models. HF_TOKEN is a *build secret* (not stored in any layer);
#    snapshots land in the HF_HOME cache mount and are reused at runtime.
COPY docker/preload_models.py ./docker/preload_models.py
RUN --mount=type=secret,id=hf_token \
    --mount=type=cache,target=/cache/huggingface \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python docker/preload_models.py || true

# Non-root runtime user; ensure cache dir is writable when volume-mounted.
RUN useradd -m -u 1000 pandaro && mkdir -p /cache && chown -R pandaro:pandaro /cache /app
USER pandaro

WORKDIR /app/backend
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/health || exit 1

CMD ["python", "-m", "pandaro.main"]
