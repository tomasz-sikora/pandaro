# Pandaro — backend

Stateless FastAPI GPU compute engine for Pandaro. Loads ASR/diarization/
paralinguistics models on demand, processes audio, streams progress over
WebSocket, and proxies LLM/embedding calls to a host Ollama. Persists nothing
between runs — the browser SPA holds the canonical, ephemeral session state.

See the repository root `README.md` for the full architecture and run
instructions.

## Quick start (dev, no GPU required)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
PANDARO_ASR_BACKEND=stub python -m pandaro.main   # serves on :9090
pytest -q
```

With `PANDARO_ASR_BACKEND=stub` (and other providers auto-falling back to stubs
when models/GPU are absent) the entire pipeline runs end-to-end for development
and testing. Install `.[ml]` on a CUDA host to enable real inference.
