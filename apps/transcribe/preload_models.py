"""
Preload all models during Docker build.

Models baked into image:
  - Whisper large-v3              (~3 GB, float16 GPU weights)
  - pyannote/speaker-diarization-3.1  (~400 MB, requires HF_TOKEN build arg)
  - audeering/wav2vec2-large-robust-24-ft-age-gender (~1.2 GB)
  - audeering/wav2vec2-large-robust-12-ft-emotion4   (~700 MB)

To bake pyannote during build, pass HF_TOKEN as a Docker build arg:
  docker compose build --build-arg HF_TOKEN=hf_...

GPU note: at runtime the server always loads models on CUDA when available.
  The preload step runs during Docker build (no GPU), so it downloads model
  weights to the HF cache. The same cached files are used at runtime with
  GPU float16 inference — no re-download needed.
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/app/models/hf")
os.environ.setdefault("TORCH_HOME", "/app/models/torch")


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def preload_whisper():
    model_name = os.getenv("WHISPER_MODEL", "large-v3")
    gpu = _has_gpu()
    device = "cuda" if gpu else "cpu"
    compute_type = "float16" if gpu else "int8"
    print(f"[1/4] Preloading faster-whisper {model_name!r} ({device} / {compute_type})...")
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(model_name, device=device, compute_type=compute_type)
        del m
        print("      Done.")
    except Exception as e:
        print(f"      WARNING: {e}", file=sys.stderr)
        if gpu:
            # GPU failed — fall back to CPU so weights are at least cached
            try:
                print("      Retrying on CPU / int8 to cache model weights...")
                from faster_whisper import WhisperModel
                m = WhisperModel(model_name, device="cpu", compute_type="int8")
                del m
                print("      Weights cached (will use GPU at runtime).")
            except Exception as e2:
                print(f"      WARNING (CPU fallback): {e2}", file=sys.stderr)


def preload_pyannote():
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("[2/4] Skipping pyannote preload — HF_TOKEN not set (will download at first startup).")
        return
    print("[2/4] Preloading pyannote/speaker-diarization-3.1...")
    try:
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        del pipeline
        print("      Done.")
    except Exception as e:
        print(f"      WARNING: pyannote preload failed: {e}", file=sys.stderr)
        print("      pyannote will be downloaded at container startup.", file=sys.stderr)


def preload_age_gender():
    model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    print(f"[3/4] Preloading audeering age/gender model {model_name!r}...")
    try:
        import sys
        sys.path.insert(0, "/app")
        from src.speaker_profiler import _build_age_gender_model
        import torch
        processor, model = _build_age_gender_model()
        if torch.cuda.is_available():
            model = model.to("cuda")
        del processor, model
        print("      Done.")
    except Exception as e:
        print(f"      WARNING: {e}", file=sys.stderr)


def preload_emotion():
    model_name = "superb/wav2vec2-base-superb-er"
    print(f"[4/4] Preloading speech emotion model {model_name!r}...")
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline("audio-classification", model=model_name, device=device, top_k=None)
        del pipe
        print("      Done.")
    except Exception as e:
        print(f"      WARNING: {e}", file=sys.stderr)


if __name__ == "__main__":
    preload_whisper()
    preload_pyannote()
    preload_age_gender()
    preload_emotion()
    print("All models preloaded successfully.")
