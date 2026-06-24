import { useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessionStore } from '../store/sessionStore'
import { useSettingsStore } from '../store/settingsStore'
import type { Segment, SpeakerProfile, AsrEngine, AgentEvent, ProcessingStep } from '@pandaro/shared-types'

function parseSegments(raw: any[], detectedLanguage: string): Segment[] {
  return raw.map((s, i) => ({
    id: Number(s.id ?? i),
    start: Number(s.start),
    end: Number(s.end),
    text: String(s.text ?? ''),
    text_pl: s.text_pl ? String(s.text_pl) : undefined,
    speaker: String(s.speaker ?? '—'),
    language: String(s.language ?? detectedLanguage),
    words: Array.isArray(s.words)
      ? s.words.map((w: any) => ({
          text: String(w.text ?? ''),
          start: Number(w.start ?? 0),
          end: Number(w.end ?? 0),
          probability: Number(w.probability ?? 1),
          alternatives: Array.isArray(w.alternatives) ? w.alternatives.map(String).filter(Boolean) : [],
        }))
      : undefined,
  }))
}

export function useAgentPipeline() {
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
    setQualityStats,
    setAgentSessionId,
    setSegmentQuality,
    setTopics,
    setQuotesAndFacts,
    setNoiseRegions,
    setSourceFile,
    appendSegments,
    applyTranslations,
    applySpeakerLabels,
    addAgentEvent,
  } = useSessionStore()
  const { settings } = useSettingsStore()
  const abortRef = useRef<AbortController | null>(null)
  // Track whether we already navigated to /transcript during streaming
  const navigatedRef = useRef(false)

  const process = useCallback(
    async (file: File, asrEngine?: AsrEngine) => {
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      const { signal } = ctrl
      navigatedRef.current = false

      const engine = asrEngine ?? settings.defaultAsrEngine
      startSession(file.name, file.size)
      setSourceFile(file)  // store file for re-process functionality

      const audioUrl = URL.createObjectURL(file)
      setAudio(audioUrl)
      setProcessing('decoding', 3, `Wysyłanie do agenta (${engine})…`)

      const form = new FormData()
      form.append('file', file)
      if (settings.sourceLanguage && settings.sourceLanguage !== 'auto') {
        form.append('language', settings.sourceLanguage)
      }
      form.append('translate', String(settings.translateToPl))
      form.append('engine', engine)
      // Pass the selected LLM model so agent uses it for all Ollama calls
      if (settings.ollamaModel) form.append('model', settings.ollamaModel)

      let res: Response
      try {
        res = await fetch(`${settings.transcribeUrl}/transcribe`, { method: 'POST', body: form, signal })
      } catch (err: any) {
        if (signal.aborted) return
        setProcessing('error', 0, '', `Błąd połączenia: ${err?.message}`)
        return
      }

      if (!res.ok) {
        const text = await res.text().catch(() => '')
        setProcessing('error', 0, '', `Serwer: ${res.status} ${text}`)
        return
      }

      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      let buf = ''

      try {
        outer: while (true) {
          const { done, value } = await reader.read()
          if (done) break
          if (signal.aborted) break
          buf += dec.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            let event: Record<string, unknown>
            try { event = JSON.parse(line.slice(6)) } catch { continue }

            const type = event.type as string

            // ── Agent activity events ───────────────────────────────────
            if (['agent_start', 'agent_thinking', 'tool_call', 'tool_result',
                 'tool_error', 'agent_memory', 'translation_quality_check',
                 'hint_injected', 'quality_report'].includes(type)) {
              // Extract session_id from agent_start
              if (type === 'agent_start' && event.session_id) {
                setAgentSessionId(String(event.session_id))
              }
              // Intercept detect_noise_regions result to populate waveform overlay
              if (type === 'tool_result' && (event.tool as string) === 'detect_noise_regions') {
                const regions = ((event.result as any)?.regions ?? []) as Array<{start_sec: number; end_sec: number; type: 'silence' | 'noise'}>
                if (regions.length > 0) setNoiseRegions(regions)
              }
              addAgentEvent(event as unknown as AgentEvent)
              continue
            }

            // ── Quality report (also handled as agent event above) ──────
            // (already included in agent activity list above)

            // ── Live segment stream (pre-diarization raw chunks) ────────
            if (type === 'segment_chunk') {
              const lang = ctx_lang.current
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              appendSegments(segs, lang)
              // Navigate to transcript page as soon as first segments arrive
              if (!navigatedRef.current && segs.length > 0) {
                navigatedRef.current = true
                navigate('/transcript')
              }
              continue
            }

            // ── Live translation stream (batch-by-batch) ────────────────
            if (type === 'translation_chunk') {
              const updates = (event.updates as any[]) ?? []
              if (updates.length > 0) {
                applyTranslations(updates.map((u: any) => ({
                  idx: Number(u.idx),
                  text_pl: String(u.text_pl ?? ''),
                })))
              }
              continue
            }

            // ── Partial segments (post-diarization preview) ─────────────
            if (type === 'partial_segments') {
              const lang = String(event.detected_language ?? 'auto')
              ctx_lang.current = lang
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              const profiles = (event.speaker_profiles as Record<string, SpeakerProfile>) ?? {}
              // Full replace — post-diarization segments have speaker labels
              setTranscript(segs, lang, /* partial= */ true)
              setSpeakerProfiles(profiles)
              if (Number(event.duration ?? 0) > 0) setDuration(Number(event.duration))
              addAgentEvent({ type: 'partial_segments' } as unknown as AgentEvent)
              if (!navigatedRef.current) {
                navigatedRef.current = true
                navigate('/transcript')
              }
              continue
            }

            // ── Diarization complete — refresh speaker labels ────────────
            if (type === 'diarization_update') {
              const lang = ctx_lang.current
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              setTranscript(segs, lang, /* partial= */ true)
              if (event.speaker_profiles) {
                setSpeakerProfiles(event.speaker_profiles as Record<string, SpeakerProfile>)
              }
              addAgentEvent({
                type: 'tool_result',
                tool: 'diarize_audio',
                result: { speaker_count: event.speaker_count },
              } as unknown as AgentEvent)
              if (!navigatedRef.current) {
                navigatedRef.current = true
                navigate('/transcript')
              }
              continue
            }

            // ── Speaker identification done — apply display names to UI ──
            if (type === 'speaker_profiles_update') {
              setSpeakerProfiles(event.speaker_profiles as Record<string, SpeakerProfile>)
              addAgentEvent({ type: 'tool_result', tool: 'identify_speakers',
                result: event.display_names as any } as unknown as AgentEvent)
              continue
            }

            // ── Segment merge/split — refresh segment list ───────────────
            if (type === 'segment_update') {
              const lang = ctx_lang.current
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              setTranscript(segs, lang, /* partial= */ true)
              addAgentEvent({
                type: 'tool_result',
                tool: String(event.operation ?? 'segment_update'),
                result: { before: event.before, after: event.after },
              } as unknown as AgentEvent)
              continue
            }

            // ── Partial summary (incremental for long recordings) ───────
            if (type === 'partial_summary') {
              const content = String(event.content ?? '')
              const wi = Number(event.window_index ?? 0)
              const wc = Number(event.window_count ?? 1)
              // Update summary with partial content so UI shows progress
              setSummary(
                `[Podsumowanie w trakcie… fragment ${wi + 1}/${wc}]\n\n${content.slice(0, 300)}`,
                content,
              )
              addAgentEvent({
                type: 'partial_summary',
                window_index: wi,
                window_count: wc,
              } as unknown as AgentEvent)
              continue
            }

            // ── Progress ────────────────────────────────────────────────
            if (type === 'progress') {
              const stage = (event.stage as string) ?? 'transcribing'
              setProcessing(stage as any, Number(event.progress ?? 0), String(event.message ?? ''))
              continue
            }

            // ── Final result ────────────────────────────────────────────
            if (type === 'result') {
              const lang = String(event.detected_language ?? 'auto')
              if (Number(event.duration ?? 0) > 0) setDuration(Number(event.duration))
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              setTranscript(segs, lang, /* partial= */ false)
              setSpeakerProfiles((event.speaker_profiles as Record<string, SpeakerProfile>) ?? {})
              if (event.entities && typeof event.entities === 'object') setEntities(event.entities as any)
              if (event.report && typeof event.report === 'string') {
                const summary = (event.summary as string) ||
                  event.report.split('\n').slice(0, 3).join(' ').slice(0, 300)
                setSummary(summary, event.report)
              }
              if (Array.isArray(event.rag_entries) && event.rag_entries.length > 0) {
                setRagEntries(event.rag_entries as any)
              }
              if (event.quality_stats && typeof event.quality_stats === 'object') {
                setQualityStats(event.quality_stats as any)
              }
              if (event.topics && Array.isArray(event.topics)) {
                setTopics(event.topics as any)
              }
              if (event.segment_quality && typeof event.segment_quality === 'object') {
                setSegmentQuality(event.segment_quality as any)
              }
              if (event.quotes_and_facts && typeof event.quotes_and_facts === 'object') {
                setQuotesAndFacts(event.quotes_and_facts as any)
              }
              addAgentEvent({ type: 'result' } as unknown as AgentEvent)
              break outer
            }

            // ── Error ───────────────────────────────────────────────────
            if (type === 'error') {
              if (!signal.aborted) setProcessing('error', 0, '', String(event.message ?? 'Błąd agenta'))
              break outer
            }
          }
        }
      } catch (err: any) {
        if (!signal.aborted) setProcessing('error', 0, '', `Błąd strumienia: ${err?.message}`)
        return
      } finally {
        reader.releaseLock()
      }

      if (!signal.aborted) {
        setProcessing('done', 100, 'Gotowe!')
        if (!navigatedRef.current) navigate('/transcript')
      }
    },
    [settings, startSession, setAudio, setDuration, setProcessing, setTranscript,
     setSpeakerProfiles, setEntities, setSummary, setRagEntries, setQualityStats,
     setAgentSessionId, setSegmentQuality, setTopics, setQuotesAndFacts, setNoiseRegions,
     setSourceFile, appendSegments, applyTranslations, applySpeakerLabels, addAgentEvent, navigate],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    // Also tell the backend to stop the running agent so it frees the GPU and
    // releases the single-analysis lock (client abort alone leaves it running).
    const sid = useSessionStore.getState().session?.agentSessionId
    const url = useSettingsStore.getState().settings.transcribeUrl
    if (sid) {
      fetch(`${url}/session/${sid}/cancel`, { method: 'POST' }).catch(() => {})
    }
  }, [])

  const reprocessFragment = useCallback(
    async (startSec: number, endSec: number, mode: 'transcription' | 'diarization' | 'translation') => {
      const session = useSessionStore.getState().session
      const file = session?.sourceFile as File | undefined
      if (!session || !file) {
        setProcessing('error', 0, '', 'Brak pliku źródłowego do ponownego przetworzenia.')
        return
      }
      if (endSec <= startSec) return

      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      const { signal } = ctrl

      const labelMap = { transcription: 'transkrypcji', diarization: 'diaryzacji', translation: 'tłumaczenia' }
      setProcessing('reprocessing', 5,
        `Ponowne przetwarzanie ${labelMap[mode]}: ${Math.round(startSec)}–${Math.round(endSec)}s…`)

      const form = new FormData()
      form.append('file', file)
      form.append('segments', JSON.stringify(session.segments))
      form.append('start_sec', String(startSec))
      form.append('end_sec', String(endSec))
      form.append('mode', mode)
      if (settings.sourceLanguage && settings.sourceLanguage !== 'auto') {
        form.append('language', settings.sourceLanguage)
      }
      if (session.detectedLanguage) form.append('detected_language', session.detectedLanguage)
      if (settings.ollamaModel) form.append('model', settings.ollamaModel)

      let res: Response
      try {
        res = await fetch(`${settings.transcribeUrl}/reprocess`, { method: 'POST', body: form, signal })
      } catch (err: any) {
        if (signal.aborted) return
        setProcessing('error', 0, '', `Błąd połączenia: ${err?.message}`)
        return
      }
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        setProcessing('error', 0, '', `Serwer: ${res.status} ${text}`)
        return
      }

      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      let buf = ''
      const lang = session.detectedLanguage || 'auto'

      try {
        outer: while (true) {
          const { done, value } = await reader.read()
          if (done || signal.aborted) break
          buf += dec.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            let event: Record<string, unknown>
            try { event = JSON.parse(line.slice(6)) } catch { continue }
            const type = event.type as string

            if (type === 'progress') {
              setProcessing((event.stage as ProcessingStep) ?? 'reprocessing', Number(event.progress ?? 0), String(event.message ?? ''))
              continue
            }
            if (type === 'translation_chunk') {
              const updates = (event.updates as any[]) ?? []
              if (updates.length > 0) {
                applyTranslations(updates.map((u: any) => ({ idx: Number(u.idx), text_pl: String(u.text_pl ?? '') })))
              }
              continue
            }
            if (type === 'diarization_update') {
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              setTranscript(segs, lang, true)
              if (event.speaker_profiles) setSpeakerProfiles(event.speaker_profiles as Record<string, SpeakerProfile>)
              continue
            }
            if (type === 'result') {
              const segs = parseSegments((event.segments as any[]) ?? [], lang)
              setTranscript(segs, lang, false)
              if (event.speaker_profiles) setSpeakerProfiles(event.speaker_profiles as Record<string, SpeakerProfile>)
              addAgentEvent({ type: 'result' } as unknown as AgentEvent)
              break outer
            }
            if (type === 'error') {
              if (!signal.aborted) setProcessing('error', 0, '', String(event.message ?? 'Błąd przetwarzania'))
              break outer
            }
          }
        }
      } catch (err: any) {
        if (!signal.aborted) setProcessing('error', 0, '', `Błąd strumienia: ${err?.message}`)
        return
      } finally {
        reader.releaseLock()
      }

      if (!signal.aborted) setProcessing('done', 100, 'Ponowne przetwarzanie zakończone.')
    },
    [settings, setProcessing, setTranscript, setSpeakerProfiles, applyTranslations, addAgentEvent],
  )

  return { process, cancel, reprocessFragment }
}

// Module-level mutable ref to track detected language during streaming
// (outside the hook because it's updated by the segment_chunk handler before
// detected_language is confirmed by the result event)
const ctx_lang = { current: 'auto' }


