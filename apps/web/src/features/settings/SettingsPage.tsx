import { useState } from 'react'
import { Save, RefreshCw, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { useSettingsStore } from '../../store/settingsStore'
import { ollamaListModels } from '../../lib/llm/ollama'

const LANGUAGES = [
  { value: 'auto', label: 'Automatyczne wykrywanie' },
  { value: 'pl', label: 'Polski' },
  { value: 'en', label: 'Angielski' },
  { value: 'ru', label: 'Rosyjski' },
  { value: 'uk', label: 'Ukraiński' },
  { value: 'de', label: 'Niemiecki' },
]

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">{label}</label>
      {hint && <p className="text-xs text-slate-400 mb-1.5">{hint}</p>}
      {children}
    </div>
  )
}

type TestStatus = 'idle' | 'testing' | 'ok' | 'error'

export default function SettingsPage() {
  const { settings, update } = useSettingsStore()
  const [saved, setSaved] = useState(false)
  const [ollamaStatus, setOllamaStatus] = useState<TestStatus>('idle')
  const [transcribeStatus, setTranscribeStatus] = useState<TestStatus>('idle')
  const [transcribeInfo, setTranscribeInfo] = useState<Record<string, string>>({})
  const [models, setModels] = useState<string[]>([])

  const save = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const testOllama = async () => {
    setOllamaStatus('testing')
    setModels([])
    try {
      const list = await ollamaListModels(settings.ollamaUrl)
      setModels(list)
      setOllamaStatus(list.length > 0 ? 'ok' : 'error')
    } catch {
      setOllamaStatus('error')
    }
  }

  const testTranscribe = async () => {
    setTranscribeStatus('testing')
    setTranscribeInfo({})
    try {
      const res = await fetch(`${settings.transcribeUrl}/health`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setTranscribeInfo(data)
      setTranscribeStatus('ok')
    } catch {
      setTranscribeStatus('error')
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b border-slate-200 px-6 py-4">
        <h1 className="font-semibold text-slate-900">Ustawienia</h1>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-xl space-y-8">

          {/* Transcription service */}
          <section>
            <h2 className="text-base font-semibold text-slate-800 mb-4">Serwis transkrypcji (backend)</h2>
            <div className="space-y-4">
              <Field
                label="URL serwisu transkrypcji"
                hint="Używaj /transcribe w Docker Compose lub http://localhost:8000 lokalnie."
              >
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={settings.transcribeUrl}
                    onChange={(e) => update({ transcribeUrl: e.target.value })}
                    className="flex-1 px-3 py-2 text-sm rounded-lg border border-slate-200 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                    placeholder="/transcribe"
                  />
                  <button
                    onClick={testTranscribe}
                    disabled={transcribeStatus === 'testing'}
                    className="px-3 py-2 text-sm rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {transcribeStatus === 'testing' ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <RefreshCw className="w-3.5 h-3.5" />
                    )}
                    Test
                  </button>
                </div>
                {transcribeStatus === 'ok' && (
                  <div className="mt-1.5 flex items-start gap-1.5 text-xs text-green-700">
                    <CheckCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-medium">Połączono.</span>{' '}
                      {transcribeInfo.asr_engine && (
                        <span>Silnik: <strong>{transcribeInfo.asr_engine}</strong>{' '}
                          ({transcribeInfo.model ?? ''})
                          {transcribeInfo.model_source && ` · ${transcribeInfo.model_source}`}
                        </span>
                      )}
                      {transcribeInfo.diarizer && (
                        <span> · diaryzacja: {transcribeInfo.diarizer}</span>
                      )}
                    </div>
                  </div>
                )}
                {transcribeStatus === 'error' && (
                  <div className="mt-1.5 flex items-center gap-1.5 text-xs text-red-600">
                    <AlertCircle className="w-3.5 h-3.5" />
                    Nie można połączyć z serwisem transkrypcji
                  </div>
                )}
              </Field>

              <Field label="Domyślny silnik ASR">
                <div className="space-y-2">
                  {[
                    { value: 'whisper', label: 'Whisper large-v3-turbo', hint: 'Szybki, mniej VRAM (~6 GB)' },
                    { value: 'vibevoice', label: 'VibeVoice-ASR 9B', hint: 'Wbudowana diaryzacja, wymaga ~18 GB VRAM' },
                  ].map((opt) => (
                    <label key={opt.value} className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="asrEngine"
                        value={opt.value}
                        checked={settings.defaultAsrEngine === opt.value}
                        onChange={() => update({ defaultAsrEngine: opt.value as any })}
                        className="mt-0.5 accent-brand-600"
                      />
                      <div>
                        <span className="text-sm text-slate-700 font-medium">{opt.label}</span>
                        <p className="text-xs text-slate-400">{opt.hint}</p>
                      </div>
                    </label>
                  ))}
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Można zmienić przed każdym nagraniem na stronie przesyłania.
                </p>
              </Field>

              <Field label="Język źródłowy" hint="Automatyczne wykrywanie jest zwykle wystarczające.">
                <select
                  value={settings.sourceLanguage}
                  onChange={(e) => update({ sourceLanguage: e.target.value as any })}
                  className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 outline-none focus:border-brand-400 bg-white"
                >
                  {LANGUAGES.map((l) => (
                    <option key={l.value} value={l.value}>{l.label}</option>
                  ))}
                </select>
              </Field>

              <Field label="Tłumaczenie">
                <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.translateToPl}
                    onChange={(e) => update({ translateToPl: e.target.checked })}
                    className="w-4 h-4 accent-brand-600"
                  />
                  Tłumacz transkrypcję na polski (via Ollama)
                </label>
                <p className="text-xs text-slate-400 mt-1">
                  Oryginał jest zawsze zachowany. Tłumaczenie jest wyświetlane jako podstawowy tekst.
                </p>
              </Field>
            </div>
          </section>

          {/* Ollama */}
          <section>
            <h2 className="text-base font-semibold text-slate-800 mb-4">Ollama / LLM</h2>
            <div className="space-y-4">
              <Field
                label="URL Ollamy"
                hint="Użyj /ollama jeśli korzystasz z Docker Compose, lub http://localhost:11434 lokalnie."
              >
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={settings.ollamaUrl}
                    onChange={(e) => update({ ollamaUrl: e.target.value })}
                    className="flex-1 px-3 py-2 text-sm rounded-lg border border-slate-200 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                    placeholder="/ollama"
                  />
                  <button
                    onClick={testOllama}
                    disabled={ollamaStatus === 'testing'}
                    className="px-3 py-2 text-sm rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 flex items-center gap-1.5 disabled:opacity-50"
                  >
                    {ollamaStatus === 'testing' ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <RefreshCw className="w-3.5 h-3.5" />
                    )}
                    Test
                  </button>
                </div>
                {ollamaStatus === 'ok' && (
                  <div className="mt-1.5 flex items-center gap-1.5 text-xs text-green-700">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Połączono. Modele: {models.slice(0, 5).join(', ')}
                    {models.length > 5 && ` +${models.length - 5}`}
                  </div>
                )}
                {ollamaStatus === 'error' && (
                  <div className="mt-1.5 flex items-center gap-1.5 text-xs text-red-600">
                    <AlertCircle className="w-3.5 h-3.5" />
                    Nie można połączyć z Ollama
                  </div>
                )}
              </Field>

              <Field label="Model do analizy (LLM)">
                <input
                  type="text"
                  value={settings.ollamaModel}
                  onChange={(e) => update({ ollamaModel: e.target.value })}
                  className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                  placeholder="gemma4:26b"
                />
                {models.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {models.map((m) => (
                      <button
                        key={m}
                        onClick={() => update({ ollamaModel: m })}
                        className="text-xs bg-slate-100 hover:bg-brand-50 hover:text-brand-700 px-2 py-0.5 rounded-full transition-colors"
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                )}
              </Field>

              <Field label="Model embeddingów">
                <input
                  type="text"
                  value={settings.ollamaEmbeddingModel}
                  onChange={(e) => update({ ollamaEmbeddingModel: e.target.value })}
                  className="w-full px-3 py-2 text-sm rounded-lg border border-slate-200 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                  placeholder="nomic-embed-text"
                />
              </Field>

              <Field label="Źródło embeddingów">
                <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.useOllamaEmbeddings}
                    onChange={(e) => update({ useOllamaEmbeddings: e.target.checked })}
                    className="w-4 h-4 accent-brand-600"
                  />
                  Używaj Ollamy do embeddingów (zalecane)
                </label>
                <p className="text-xs text-slate-400 mt-1">
                  Odznacz, aby używać lokalnego modelu w przeglądarce (Xenova/all-MiniLM-L6-v2, ~22 MB).
                </p>
              </Field>
            </div>
          </section>

          {/* Save */}
          <button
            onClick={save}
            className="flex items-center gap-2 px-5 py-2.5 bg-brand-600 text-white rounded-xl text-sm font-medium hover:bg-brand-700 transition-colors"
          >
            {saved ? (
              <><CheckCircle className="w-4 h-4" />Zapisano</>
            ) : (
              <><Save className="w-4 h-4" />Zapisz ustawienia</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
