# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/hf
ENV TORCH_HOME=/app/models/torch

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg curl git libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure python/pip point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch with CUDA 12.4 first (separate layer for cache efficiency)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# bitsandbytes looks for libnvJitLink.so.13; it ships in nvidia-nvjitlink-cu13
# which is installed as a dependency of PyTorch CUDA packages.
# Expose it via LD_LIBRARY_PATH so bitsandbytes can find it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-nvjitlink-cu12 2>/dev/null || true
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}

# Optional: flash-attn for faster VibeVoice inference on CUDA
# Build takes ~10 min; set SKIP_FLASH_ATTN=true to skip
ARG SKIP_FLASH_ATTN=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$SKIP_FLASH_ATTN" != "true" ]; then \
    pip install flash-attn --no-build-isolation || \
    echo "flash-attn build failed, continuing without it (sdpa will be used)"; \
    fi

# Install VibeVoice package from official GitHub repo (local weights, no Azure)
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/microsoft/VibeVoice.git /opt/VibeVoice && \
    pip install /opt/VibeVoice
# VibeVoice wheel installs metadata-only; make source importable via PYTHONPATH
ENV PYTHONPATH=/opt/VibeVoice:${PYTHONPATH}

# Install NVIDIA NeMo ASR toolkit (Nemotron 3.5 ASR engine)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython packaging && \
    git clone --depth 1 https://github.com/NVIDIA/NeMo.git /opt/NeMo && \
    pip install \
        "aistore" "fsspec>=2024.12.0" "huggingface_hub>=0.24" \
        "cuda-bindings" "numpy>=1.22" "onnx>=1.7.0" "scikit-learn" \
        "setuptools>=70.0.0" "smart-open" "tensorboard" "text-unidecode" \
        "torch>=2.6.0" "tqdm>=4.41.0" "wrapt" \
        "hydra-core>1.3,<=1.3.2" "lightning>2.2.1,<=2.4.0" "omegaconf<=2.3" \
        "torchmetrics>=0.11.0" "transformers" "wandb" "webdataset>=0.2.86" \
        "nv_one_logger_core>=2.3.1" "nv_one_logger_training_telemetry>=2.3.1" \
        "nv_one_logger_pytorch_lightning_integration>=2.3.1" \
        "braceexpand" "einops" "kaldialign" "lhotse>=1.33.0" \
        "librosa>=0.10.1" "sacrebleu" "soundfile" "whisper_normalizer" && \
    pip install /opt/NeMo && \
    rm -rf /opt/NeMo/.git

# Install app dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Model preload ─────────────────────────────────────────────────────────────
# IMPORTANT: preload_models.py is copied and run BEFORE src/ so that any change
# to application source code does NOT invalidate this cache layer.
# This layer is only re-run when requirements.txt or preload_models.py changes.
#
# BuildKit cache mounts:
#   /root/.cache/huggingface  – HF metadata, tokenizers, configs (small, reused across builds)
#   /app/models/hf            – actual model weights baked into image layer
#     → on first container start with an empty named volume, Docker seeds the
#       volume from this image layer automatically (named-volume copy-on-first-use).
#     → on subsequent starts the volume already has the weights — instant load.
COPY preload_models.py .

ARG PRELOAD_MODELS=true
ARG DOWNLOAD_VIBEVOICE=false
ARG HF_TOKEN=""
ENV DOWNLOAD_VIBEVOICE=${DOWNLOAD_VIBEVOICE}

RUN --mount=type=cache,target=/root/.cache/huggingface \
    echo "PRELOAD_MODELS=${PRELOAD_MODELS}" && \
    if [ "$PRELOAD_MODELS" = "true" ]; then \
        HF_TOKEN="$HF_TOKEN" python preload_models.py; \
    fi

# ── Application source (changes here never invalidate the preload layer) ──────
COPY src/ ./src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
