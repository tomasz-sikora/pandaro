# Heimdall — Service Specification

## 1. Concept

Heimdall is a **private, self-hosted audio intelligence platform** for analysing recorded conversations.

A user uploads an audio file and receives, in return, a fully structured analysis: who spoke, what was said, the emotional tone, a Polish translation, extracted entities, a summary, and an interactive Q&A interface grounded in the recording.

All processing runs locally — no data leaves the user's infrastructure. There is no user authentication, no multi-tenancy, and no persistent storage. Each browser session is independent and ephemeral.

---

## 2. System Overview

```
Browser (SPA)
  │ upload file
  ▼
Transcription Service (Python / GPU)
  │ segments + speaker profiles
  ▼
Browser — post-processing (in-page)
  │ LLM calls
  ▼
Ollama (local LLM / embedding server)
```

The system has three runtime components:

| Component | Role |
|-----------|------|
| **Web SPA** | User interface, orchestration, RAG, chat |
| **Transcription Service** | ASR, diarization, speaker profiling, translation |
| **Ollama** | LLM inference (entity extraction, summary, chat, embeddings) |

---

## 3. Processing Pipeline

Processing is a linear, ordered pipeline triggered by a single file upload. Each stage streams progress to the UI in real time via Server-Sent Events (SSE).

### Stage 1 — Audio decoding
- Accept any common audio/video format (MP3, MP4, M4A, WAV, and others).
- Normalise to 16 kHz mono PCM for all downstream processing.

### Stage 2 — Speech-to-text (ASR)
- Transcribe speech to text with word-level timestamps.
- Detect the spoken language automatically, or accept a user-specified language hint.
- Three selectable ASR engines (user choice at upload time):
  - **Whisper** (`faster-whisper large-v3-turbo`) — default; fast and accurate.
  - **VibeVoice-ASR** (`microsoft/VibeVoice-ASR`, 9 B) — single-pass ASR + diarization, 50+ languages.
  - **Nemotron 3.5 ASR** (`nvidia/nemotron-3.5-asr-streaming-0.6b`, 600 M) — cache-aware FastConformer-RNNT, 40 language-locales.

### Stage 3 — Speaker diarization
- Identify individual speakers and assign each transcript segment to a speaker label (`GŁOS_01`, `GŁOS_02`, …).
- VibeVoice-ASR performs diarization internally; Whisper and Nemotron use a separate pyannote diarizer.

### Stage 4 — Speaker profiling
- For each identified speaker, estimate:
  - **Gender** (female / male) with confidence score.
  - **Age group** (child / young / adult / senior) with numeric age estimate.
  - **Dominant emotion** (e.g. happy, sad, neutral, angry) with per-class probabilities.
  - **Speech rate** (syllables per second, labelled slow/normal/fast).
  - **Signal-to-noise ratio** (dB, labelled clean/moderate/noisy).

### Stage 5 — Translation to Polish
- If the detected language is not Polish and the user has translation enabled, translate all segments into Polish using a local Ollama LLM.
- The original text is preserved alongside the translation.

### Stage 6 — Entity extraction
- Extract structured named entities from the full transcript text using a local Ollama LLM.
- Output categories: **persons**, **organisations**, **locations**, **dates**, **keywords**.

### Stage 7 — Embeddings & RAG index
- Split the transcript into overlapping text chunks.
- Embed each chunk using either a local Ollama embedding model or an in-browser WASM model.
- Store all chunks with their embeddings and segment metadata in an in-memory vector index.

### Stage 8 — Summary
- Generate a free-text summary and a longer structured report from the full transcript using a local Ollama LLM.

---

## 4. Functional Requirements

### 4.1 Upload
- Accept drag-and-drop or click-to-browse file selection.
- Supported formats: MP3, MP4, M4A, WAV (and generically `audio/*`, `video/mp4`).
- The user selects the ASR engine before starting.
- Display real-time progress with labelled stages and a percentage bar.
- Allow cancellation of an in-progress pipeline.

### 4.2 Transcript View
- Display all transcript segments in chronological order.
- Each segment shows: timestamp (start), speaker label, original text, Polish translation (if different).
- Speaker labels are colour-coded consistently across the view.

### 4.3 Analysis View
- **Speaker profiles**: per-speaker cards showing gender, age, emotion, speech rate, and SNR.
- **Entities**: tabbed or grouped display of extracted persons, organisations, locations, dates, and keywords.
- **Summary**: short summary paragraph and full report.
- Audio duration and detected language shown.

### 4.4 Chat
- Conversational Q&A interface grounded exclusively in the content of the current recording.
- Each assistant answer cites the source segments it drew from (timestamp + speaker).
- Clicking a citation opens a side panel showing the surrounding transcript context.
- The chat uses retrieval-augmented generation: the question is embedded, the nearest transcript chunks are retrieved, and a local LLM generates the answer with those chunks as context.
- Full conversation history is maintained within the session.

### 4.5 Settings
- Configurable parameters (persisted in browser local storage):
  - Transcription service URL.
  - Ollama base URL.
  - Ollama chat model.
  - Ollama embedding model.
  - Toggle: use Ollama embeddings vs. in-browser WASM embeddings.
  - Default ASR engine.
  - Source language hint (auto-detect or fixed language).
  - Toggle: translate to Polish.
- Live connectivity test for both the transcription service and Ollama.
- Discovery of available Ollama models.

### 4.6 Session lifecycle
- One active session at a time per browser tab.
- All session data is held in memory; nothing is written to disk or a database.
- Starting a new upload clears the previous session.
- Navigating away from the tab loses all data.

---

## 5. ASR Engine Selection

| Engine | Size | Languages | Diarization | GPU VRAM | Load strategy |
|--------|------|-----------|-------------|----------|---------------|
| Whisper `large-v3-turbo` | ~1.5 GB | 100+ | pyannote (separate) | ~3 GB | Loaded at startup |
| VibeVoice-ASR | ~18 GB | 50+ | Built-in | ~18 GB | Lazy — loaded on first use, unloaded after |
| Nemotron 3.5 ASR | ~1.2 GB | 40 locales | pyannote (separate) | ~1.2 GB | Lazy — loaded on first use, stays loaded |

Only one engine occupies the GPU at a time. The backend serialises all inference requests on a single executor thread.

---

## 6. Language Support

- Input language: auto-detected or user-specified.
- Output language: configurable — translation to Polish is optional.
- ASR engine choice determines supported input language set (see table above).
- Entity extraction and summary are generated in the detected/original language.

---

## 7. Deployment

- Packaged as Docker Compose with three services: `transcribe`, `proxy`, `web`.
- GPU (NVIDIA CUDA) is required for the transcription service.
- Ollama runs separately on the host (or another machine) and is accessed via a configurable URL.
- All model weights are downloaded to a persistent volume on first use.
- No external API calls — fully air-gappable once weights are downloaded.
