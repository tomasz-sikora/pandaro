import { create } from 'zustand'
import type {
  Session,
  Segment,
  Entities,
  VectorEntry,
  ChatMessage,
  ProcessingStep,
  SpeakerProfile,
  AgentEvent,
  QualityStats,
  QuotesAndFacts,
} from '@pandaro/shared-types'

interface SessionStore {
  session: Session | null
  startSession: (fileName: string, fileSize: number) => void
  setAudio: (url: string) => void
  setDuration: (duration: number) => void
  setProcessing: (step: ProcessingStep, progress: number, message: string, error?: string) => void
  setTranscript: (segments: Segment[], detectedLanguage: string, partial?: boolean) => void
  setSpeakerProfiles: (profiles: Record<string, SpeakerProfile>) => void
  setEntities: (entities: Entities) => void
  setSummary: (summary: string, report: string) => void
  setRagEntries: (entries: VectorEntry[]) => void
  setQualityStats: (stats: QualityStats) => void
  setAgentSessionId: (id: string) => void
  setSegmentQuality: (quality: Record<number, number>) => void
  setTopics: (topics: Session['topics']) => void
  setQuotesAndFacts: (qf: QuotesAndFacts) => void
  appendSegments: (segs: Segment[], detectedLanguage?: string) => void
  applyTranslations: (updates: Array<{ idx: number; text_pl: string }>) => void
  applySpeakerLabels: (segments: Segment[]) => void
  addChatMessage: (msg: ChatMessage) => void
  updateLastAssistantMessage: (content: string, sources?: ChatMessage['sources']) => void
  updateSegmentText: (segmentId: number, text: string) => void
  updateSegmentWord: (segmentId: number, wordIdx: number, altText: string) => void
  loadTranscript: (fileName: string, segments: Segment[], detectedLanguage?: string) => void
  clearSession: (transcribeUrl?: string) => void
  addAgentEvent: (event: AgentEvent) => void
}

function makeId() {
  return Math.random().toString(36).slice(2, 10)
}

export const useSessionStore = create<SessionStore>()((set) => ({
  session: null,

  startSession: (fileName, fileSize) =>
    set({
      session: {
        id: makeId(),
        fileName,
        fileSize,
        duration: null,
        detectedLanguage: null,
        processing: { step: 'decoding', progress: 0, message: 'Odczytywanie pliku…' },
        segments: [],
        speakerProfiles: {},
        entities: null,
        summary: null,
        report: null,
        ragEntries: [],
        chat: [],
        audioObjectUrl: null,
        agentEvents: [],
        segmentsPartial: false,
        qualityStats: null,
        agentSessionId: null,
        segmentQuality: {},
        topics: [],
        quotesAndFacts: null,
        createdAt: Date.now(),
      },
    }),

  setAudio: (url) =>
    set((s) => ({
      session: s.session ? { ...s.session, audioObjectUrl: url } : null,
    })),

  setDuration: (duration) =>
    set((s) => ({
      session: s.session ? { ...s.session, duration } : null,
    })),

  setProcessing: (step, progress, message, error) =>
    set((s) => ({
      session: s.session
        ? { ...s.session, processing: { step, progress, message, error } }
        : null,
    })),

  setTranscript: (segments, detectedLanguage, partial = false) =>
    set((s) => ({
      session: s.session ? { ...s.session, segments, detectedLanguage, segmentsPartial: partial } : null,
    })),

  setSpeakerProfiles: (speakerProfiles) =>
    set((s) => ({
      session: s.session ? { ...s.session, speakerProfiles } : null,
    })),

  setEntities: (entities) =>
    set((s) => ({
      session: s.session ? { ...s.session, entities } : null,
    })),

  setSummary: (summary, report) =>
    set((s) => ({
      session: s.session ? { ...s.session, summary, report } : null,
    })),

  setRagEntries: (ragEntries) =>
    set((s) => ({
      session: s.session ? { ...s.session, ragEntries } : null,
    })),

  setQualityStats: (qualityStats) =>
    set((s) => ({
      session: s.session ? { ...s.session, qualityStats } : null,
    })),

  appendSegments: (segs, detectedLanguage) =>
    set((s) => {
      if (!s.session) return {}
      // De-duplicate by id — keep existing if already present
      const existingIds = new Set(s.session.segments.map((seg) => seg.id))
      const fresh = segs.filter((seg) => !existingIds.has(seg.id))
      if (fresh.length === 0) return {}
      return {
        session: {
          ...s.session,
          segments: [...s.session.segments, ...fresh],
          segmentsPartial: true,
          detectedLanguage: detectedLanguage ?? s.session.detectedLanguage,
        },
      }
    }),

  applyTranslations: (updates) =>
    set((s) => {
      if (!s.session || s.session.segments.length === 0) return {}
      const segments = s.session.segments.map((seg, i) => {
        const upd = updates.find((u) => u.idx === i)
        return upd ? { ...seg, text_pl: upd.text_pl } : seg
      })
      return { session: { ...s.session, segments } }
    }),

  applySpeakerLabels: (updated) =>
    set((s) => {
      if (!s.session) return {}
      // Rebuild by id lookup from the updated array
      const byId = new Map(updated.map((seg) => [seg.id, seg]))
      const segments = s.session.segments.map((seg) => {
        const u = byId.get(seg.id)
        return u ? { ...seg, speaker: u.speaker } : seg
      })
      return { session: { ...s.session, segments } }
    }),

  addChatMessage: (msg) =>
    set((s) => ({
      session: s.session
        ? { ...s.session, chat: [...s.session.chat, msg] }
        : null,
    })),

  updateLastAssistantMessage: (content, sources) =>
    set((s) => {
      if (!s.session) return {}
      const chat = [...s.session.chat]
      const lastIdx = chat.length - 1
      if (lastIdx >= 0 && chat[lastIdx].role === 'assistant') {
        chat[lastIdx] = { ...chat[lastIdx], content, sources }
      }
      return { session: { ...s.session, chat } }
    }),

  clearSession: (transcribeUrl?: string) =>
    set((s) => {
      if (s.session?.audioObjectUrl) {
        URL.revokeObjectURL(s.session.audioObjectUrl)
      }
      // Clear backend cache so changing model starts fresh
      if (transcribeUrl) {
        fetch(`${transcribeUrl}/cache`, { method: 'DELETE' }).catch(() => {})
      }
      return { session: null }
    }),

  setAgentSessionId: (agentSessionId) =>
    set((s) => ({
      session: s.session ? { ...s.session, agentSessionId } : null,
    })),

  setSegmentQuality: (segmentQuality) =>
    set((s) => ({
      session: s.session ? { ...s.session, segmentQuality } : null,
    })),

  setTopics: (topics) =>
    set((s) => ({
      session: s.session ? { ...s.session, topics } : null,
    })),

  setQuotesAndFacts: (quotesAndFacts) =>
    set((s) => ({
      session: s.session ? { ...s.session, quotesAndFacts } : null,
    })),

  updateSegmentText: (segmentId, text) =>
    set((s) => {
      if (!s.session) return {}
      const segments = s.session.segments.map((seg) =>
        seg.id === segmentId ? { ...seg, text, text_pl: text, alternatives: seg.alternatives } : seg,
      )
      return { session: { ...s.session, segments } }
    }),

  updateSegmentWord: (segmentId, wordIdx, altText) =>
    set((s) => {
      if (!s.session) return {}
      const segments = s.session.segments.map((seg) => {
        if (seg.id !== segmentId || !seg.words) return seg
        const words = seg.words.map((w, i) => {
          if (i !== wordIdx) return w
          // preserve leading space Whisper attaches to each word
          const prefix = /^\s+/.exec(w.text)?.[0] ?? ''
          return { ...w, text: prefix + altText.trim(), alternatives: [] }
        })
        const reconstructed = words.map((w) => w.text).join('').trim()
        return { ...seg, words, text: reconstructed, text_pl: reconstructed }
      })
      return { session: { ...s.session, segments } }
    }),

  loadTranscript: (fileName, segments, detectedLanguage = 'auto') =>
    set({
      session: {
        id: makeId(),
        fileName,
        fileSize: 0,
        duration: segments.length > 0 ? segments[segments.length - 1].end : null,
        detectedLanguage,
        processing: { step: 'done', progress: 100, message: 'Załadowano z pliku' },
        segments,
        speakerProfiles: {},
        entities: null,
        summary: null,
        report: null,
        ragEntries: [],
        chat: [],
        agentEvents: [],
        createdAt: Date.now(),
      },
    }),

  addAgentEvent: (event) =>
    set((s) => ({
      session: s.session
        ? { ...s.session, agentEvents: [...s.session.agentEvents, event] }
        : null,
    })),
}))
