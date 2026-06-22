import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Settings } from '@heimdall/shared-types'

interface SettingsStore {
  settings: Settings
  update: (patch: Partial<Settings>) => void
}

const defaults: Settings = {
  transcribeUrl: '/transcribe',
  ollamaUrl: '/ollama',
  ollamaModel: 'ministral-3:14b',
  ollamaEmbeddingModel: 'embeddinggemma',
  useOllamaEmbeddings: true,
  whisperModel: 'large-v3',
  sourceLanguage: 'auto',
  translateToPl: true,
  defaultAsrEngine: 'whisper',
  theme: 'light',
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      settings: defaults,
      update: (patch) =>
        set((s) => ({ settings: { ...s.settings, ...patch } })),
    }),
    { name: 'heimdall-settings' },
  ),
)
