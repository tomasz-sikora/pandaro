# Decisions

Chronological record of significant architectural and product decisions.
Newest first.

## 2026-06 — Backend cleanup, single-analysis, fragment re-processing

### Keep the hand-rolled agent loop (do not adopt an external agent SDK)
The processing pipeline is driven by a custom Ollama tool-calling loop in
`apps/transcribe/src/agent/__init__.py`. We evaluated moving to an external agent
SDK (e.g. Docker agent SDK) for Ollama orchestration. **Decision: keep the custom
loop.** It is small, fully observable (every tool call is streamed over SSE),
already integrated with our single-GPU executor, hint injection and cancellation,
and adds zero dependencies. An SDK would add abstraction without solving a problem
we actually have.

### Remove Nemotron
The Nemotron transcriber path was unused and pulled in a heavy NeMo dependency.
Removed from `main.py`, `tools.py`, `prompts.py`, `shared-types`, the Dockerfile and
`requirements.txt`. ASR engines are now `whisper` (default) and `vibevoice` (lazy).

### Single analysis at a time (HTTP 409)
We run on a single GPU with a `ThreadPoolExecutor(max_workers=1)`. Concurrent
`/transcribe` or `/reprocess` requests are now rejected with **HTTP 409** via
`is_busy()` instead of silently queueing behind the executor. The UI surfaces a
clear "an analysis is already running" message.

### Progress tracking + cancellation
Progress is already streamed as SSE `progress` events. Added
`POST /session/{id}/cancel`, which sets `ctx.cancelled`; the agent loop checks the
flag between steps and inside long transcription batches and aborts cleanly,
emitting a `cancelled` event. The frontend `cancel()` both aborts the fetch and
calls the cancel endpoint so the backend releases the GPU and the single-analysis
lock.

### Remove one-example "magic" tuning
Several heuristics had been hard-coded from a single sample recording and were
removed in favour of signal-based logic:
- `HALLUCINATION_PHRASES` curated list → signal-only hallucination filter
  (confidence + no-speech probability + words-per-second).
- `KEEP_SEPARATE` interjection word list in `merge_short_segments` → gap-only merge.
- Phone-call dynamic-range threshold switching in `merge_duplicate_speakers` →
  a single conservative default.

### Conservative speaker merging (safety floor 0.99)
`merge_duplicate_speakers` is destructive: it collapses speaker labels. With the
previous 0.97 MFCC-cosine threshold the agent merged genuinely different phone-call
voices (similarity ~0.98 from codec compression), collapsing the 5-speaker test
recording to 3. **Decision: default and hard floor of 0.99**, plus prompt guidance
to call the tool only when pyannote has clearly over-split one speaker. This keeps
the agent in control (no per-recording magic) while preventing the regression. The
existing "all pairwise similarities > 0.99 → skip" guard remains.

### Backend owns all processing; fragment re-processing
All transcription/diarization/translation runs on the backend (the browser-WASM
path described in the old README is retired). Added `POST /reprocess` so a user can
select a time range in the UI timeline (drag-select on the waveform) and re-run just
one step on that fragment:
- **transcription** — re-run Whisper on the range, keep speakers by time-overlap.
- **diarization** — re-run pyannote on the fragment, remap labels onto existing
  speakers via MFCC cosine (≥ 0.75) so no phantom speakers are introduced.
- **translation** — re-translate the in-range segments.

The frontend keeps the source audio and current segments and posts them with the
range and mode; the result is spliced back and streamed as a fresh `result` event.
