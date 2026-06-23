import { useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessionStore } from '../store/sessionStore'
import { useSettingsStore } from '../store/settingsStore'
import { chunkSegments, buildVectorEntries } from '../lib/rag/chunker'
import { ollamaComplete, ollamaEmbed } from '../lib/llm/ollama'
import {
  entityExtractionPrompt,
  summaryPrompt,
  summaryReducePrompt,
} from '../lib/llm/prompts'
import type { Segment, SpeakerProfile, AsrEngine } from '@pandaro/shared-types'

// Maximum number of chunks to embed in one pass
const MAX_EMBED_CHUNKS = 300

// Characters per LLM window for NER / summary (stays well under 4K token models)
const LLM_WINDOW_CHARS = 6_000
// Max windows to process (caps total LLM time on very long recordings)
const MAX_NER_WINDOWS = 8
const MAX_SUMMARY_WINDOWS = 6

/**
 * Calls the Python transcription backend via Server-Sent Events (SSE).
 * The backend handles: audio decoding, Whisper/VibeVoice, diarization,
 * speaker profiling, and Polish translation.
 */
async function* streamTranscription(
  file: File,
  transcribeUrl: string,
  sourceLanguage: string,
  translateToPl: boolean,
  asrEngine: AsrEngine,
  signal: AbortSignal,
): AsyncGenerator<Record<string, unknown>> {
  const form = new FormData()
  form.append('file', file)
  if (sourceLanguage && sourceLanguage !== 'auto') {
    form.append('language', sourceLanguage)
  }
  form.append('translate', String(translateToPl))
  form.append('engine', asrEngine)

  const res = await fetch(`${transcribeUrl}/transcribe`, {
    method: 'POST',
    body: form,
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Serwer transkrypcji: ${res.status} ${text}`)
  }

  const reader = res.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          yield JSON.parse(line.slice(6))
        } catch { /* skip malformed */ }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/** Run embeddings in the local Worker, always terminating it on finish/error. */
function embedInWorker(texts: string[]): Promise<number[][]> {
  return new Promise<number[][]>((resolve, reject) => {
    const worker = new Worker(
      new URL('../workers/embedding.worker.ts', import.meta.url),
      { type: 'module' },
    )

    const cleanup = () => worker.terminate()

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === 'result') {
        cleanup()
        resolve(msg.embeddings as number[][])
      } else if (msg.type === 'error') {
        cleanup()
        reject(new Error(msg.message as string))
      }
      // 'progress' messages are informational — ignore here
    }

    worker.onerror = (e) => {
      cleanup()
      reject(new Error(e.message))
    }

    worker.postMessage({ type: 'embed', texts })
  })
}

/**
 * Split a long text into overlapping windows for chunked LLM processing.
 * Each window is at most `windowChars` characters; windows overlap by ~10%.
 */
function splitIntoWindows(text: string, windowChars: number, maxWindows: number): string[] {
  if (text.length <= windowChars) return [text]
  const overlap = Math.floor(windowChars * 0.1)
  const step = windowChars - overlap
  const windows: string[] = []
  for (let i = 0; i < text.length && windows.length < maxWindows; i += step) {
    windows.push(text.slice(i, i + windowChars))
  }
  return windows
}

/**
 * Merge multiple entity extraction results, deduplicating each list.
 */
function mergeEntities(results: Array<Record<string, string[]>>) {
  const merged: Record<string, Set<string>> = {
    persons: new Set(), organizations: new Set(), locations: new Set(),
    dates: new Set(), keywords: new Set(),
  }
  for (const r of results) {
    for (const key of Object.keys(merged)) {
      for (const v of (r[key] ?? [])) {
        if (v) merged[key].add(v.trim())
      }
    }
  }
  return {
    persons: [...merged.persons],
    organizations: [...merged.organizations],
    locations: [...merged.locations],
    dates: [...merged.dates],
    keywords: [...merged.keywords].slice(0, 20),
  }
}

/**
 * Build RAG embeddings for the given segments using Ollama.
 */
export async function computeRagEntries(
  segments: import('@pandaro/shared-types').Segment[],
  cfg: { baseUrl: string; model: string; embeddingModel: string },
  _useOllama?: boolean,
): Promise<import('@pandaro/shared-types').VectorEntry[]> {
  const chunks = chunkSegments(segments)
  const safeChunks = chunks.slice(0, MAX_EMBED_CHUNKS)
  const chunkTexts = safeChunks.map((c) => c.text)
  const embeddings = await ollamaEmbed(cfg, chunkTexts)
  return buildVectorEntries(safeChunks, embeddings)
}

export function useProcessingPipeline() {
  const navigate = useNavigate()
  const {
    startSession,
    setAudio,
    setDuration,
    setProcessing,
    setTranscript,
    setSpeakerProfiles,
    setEntities,
    setSummary,
    setRagEntries,
  } = useSessionStore()
  const { settings } = useSettingsStore()

  // Ref so cancellation works across renders
  const abortRef = useRef<AbortController | null>(null)

  const process = useCallback(
    async (file: File, asrEngine?: AsrEngine) => {
      // Cancel any previous in-flight pipeline
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      const { signal } = ctrl

      const engine = asrEngine ?? settings.defaultAsrEngine
      startSession(file.name, file.size)

      // Create a persistent object URL for in-browser playback (revoked on clearSession)
      const audioUrl = URL.createObjectURL(file)
      setAudio(audioUrl)

      setProcessing('decoding', 3, `Wysyłanie do serwera transkrypcji (${engine})…`)

      const cfg = {
        baseUrl: settings.ollamaUrl,
        model: settings.ollamaModel,
        embeddingModel: settings.ollamaEmbeddingModel,
      }

      // ── 1. Stream transcription from backend ──────────────────────────────
      let segments: Segment[] = []
      let detectedLanguage = 'auto'
      let speakerProfilesRaw: Record<string, SpeakerProfile> = {}

      try {
        for await (const event of streamTranscription(
          file,
          settings.transcribeUrl,
          settings.sourceLanguage,
          settings.translateToPl,
          engine,
          signal,
        )) {
          if (signal.aborted) return
          if (event.type === 'progress') {
            const step = (event.stage as string) ?? 'transcribing'
            setProcessing(
              step as any,
              Number(event.progress ?? 0),
              String(event.message ?? ''),
            )
          } else if (event.type === 'result') {
            detectedLanguage = String(event.detected_language ?? 'auto')
            const dur = Number(event.duration ?? 0)
            if (dur > 0) setDuration(dur)

            segments = ((event.segments as any[]) ?? []).map((s, i) => ({
              id: i,
              start: Number(s.start),
              end: Number(s.end),
              text: String(s.text ?? ''),
              text_pl: s.text_pl ? String(s.text_pl) : undefined,
              speaker: String(s.speaker ?? `GŁOS_01`),
              language: String(s.language ?? detectedLanguage),
              words: Array.isArray(s.words) ? s.words.map((w: any) => ({
                text: String(w.text ?? ''),
                start: Number(w.start ?? 0),
                end: Number(w.end ?? 0),
                probability: Number(w.probability ?? 1),
                alternatives: Array.isArray(w.alternatives) ? w.alternatives.map(String).filter(Boolean) : [],
              })) : undefined,
              alternatives: Array.isArray(s.alternatives) ? s.alternatives.map(String).filter(Boolean) : undefined,
            }))
            speakerProfilesRaw = (event.speaker_profiles as any) ?? {}
            setTranscript(segments, detectedLanguage)
            setSpeakerProfiles(speakerProfilesRaw)
          } else if (event.type === 'error') {
            throw new Error(String(event.message ?? 'Błąd serwera transkrypcji'))
          }
        }
      } catch (err: any) {
        if (signal.aborted) return
        setProcessing('error', 0, '', `Błąd transkrypcji: ${err?.message}`)
        return
      }

      if (segments.length === 0) {
        setProcessing('error', 0, '', 'Transkrypcja nie zwróciła żadnych segmentów.')
        return
      }

      // Build full text for LLM — no hard truncation; we window it per-call
      const fullText = segments
        .map((s) => s.text_pl ?? s.text)
        .join(' ')

      const durationMin = Math.round((segments[segments.length - 1]?.end ?? 0) / 60)
      const isLong = durationMin > 30

      // ── 2. Entity extraction (chunked for long recordings) ────────────────
      if (!signal.aborted) {
        try {
          setProcessing('extracting', 80, `Ekstrakcja encji${isLong ? ` (${durationMin} min nagranie, kilka okien)` : ''}…`)
          const windows = splitIntoWindows(fullText, LLM_WINDOW_CHARS, MAX_NER_WINDOWS)
          const entityResults: Array<Record<string, string[]>> = []

          for (let wi = 0; wi < windows.length && !signal.aborted; wi++) {
            if (windows.length > 1) {
              setProcessing('extracting', 80 + Math.round((wi / windows.length) * 5),
                `Ekstrakcja encji okno ${wi + 1}/${windows.length}…`)
            }
            const raw = await ollamaComplete(
              cfg,
              entityExtractionPrompt(windows[wi], detectedLanguage),
              signal,
              180_000,
            )
            const jsonMatch = raw.match(/\{[\s\S]*\}/)
            if (jsonMatch) {
              try { entityResults.push(JSON.parse(jsonMatch[0])) } catch { /* skip */ }
            }
          }

          if (!signal.aborted && entityResults.length > 0) {
            setEntities(mergeEntities(entityResults))
          }
        } catch { /* non-fatal */ }
      }

      // ── 3. Embeddings + RAG ───────────────────────────────────────────────
      if (!signal.aborted) {
        setProcessing('embedding', 88, 'Budowanie indeksu RAG…')
        try {
          const entries = await computeRagEntries(segments, cfg, settings.useOllamaEmbeddings)
          if (!signal.aborted) setRagEntries(entries)
        } catch (err) {
          console.warn('[RAG] Embedding failed entirely:', err)
        }
      }

      // ── 4. Summary (chunked map-reduce for long recordings) ───────────────
      if (!signal.aborted) {
        try {
          const windows = splitIntoWindows(fullText, LLM_WINDOW_CHARS, MAX_SUMMARY_WINDOWS)
          setProcessing('summarizing', 93, `Generowanie podsumowania${windows.length > 1 ? ` (${windows.length} fragmentów)` : ''}…`)

          if (windows.length === 1) {
            // Short recording — single call
            const report = await ollamaComplete(cfg, summaryPrompt(windows[0], detectedLanguage), signal, 300_000)
            if (!signal.aborted) {
              setSummary(report.split('\n').slice(0, 3).join(' ').slice(0, 300), report)
            }
          } else {
            // Long recording — map: summarise each window, reduce: final synthesis
            const partials: string[] = []
            for (let wi = 0; wi < windows.length && !signal.aborted; wi++) {
              setProcessing('summarizing', 93 + Math.round((wi / windows.length) * 5),
                `Streszczenie fragment ${wi + 1}/${windows.length}…`)
              const partial = await ollamaComplete(cfg, summaryPrompt(windows[wi], detectedLanguage), signal, 300_000)
              if (partial) partials.push(partial)
            }

            if (!signal.aborted && partials.length > 0) {
              setProcessing('summarizing', 98, 'Łączenie streszczeń…')
              const combinedPartials = partials
                .map((p, i) => `### Fragment ${i + 1}\n${p}`)
                .join('\n\n')
                .slice(0, LLM_WINDOW_CHARS * 2)
              const finalReport = await ollamaComplete(
                cfg,
                summaryReducePrompt(combinedPartials, durationMin),
                signal,
                360_000,
              )
              if (!signal.aborted) {
                setSummary(finalReport.split('\n').slice(0, 3).join(' ').slice(0, 300), finalReport)
              }
            }
          }
        } catch { /* non-fatal */ }
      }



      if (!signal.aborted) {
        setProcessing('done', 100, 'Gotowe!')
        navigate('/transcript')
      }
    },
    [
      settings,
      startSession,
      setDuration,
      setProcessing,
      setTranscript,
      setSpeakerProfiles,
      setEntities,
      setSummary,
      setRagEntries,
      navigate,
    ],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  return { process, cancel }
}

