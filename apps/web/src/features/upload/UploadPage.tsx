import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Mic, AlertCircle, Loader2, FileJson } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import { useProcessingPipeline } from '../../hooks/useProcessingPipeline'
import type { Segment } from '@heimdall/shared-types'

const ACCEPTED = ['.mp3', '.mp4', '.m4a', '.wav', 'audio/*', 'video/mp4']
const STEPS = [
  { key: 'decoding',      label: 'Wysyłanie do serwera' },
  { key: 'loading_model', label: 'Ładowanie modelu ASR' },
  { key: 'transcribing',  label: 'Transkrypcja (GPU)' },
  { key: 'diarizing',     label: 'Identyfikacja mówców' },
  { key: 'profiling',     label: 'Analiza cech mówców' },
  { key: 'translating',   label: 'Tłumaczenie na polski' },
  { key: 'extracting',    label: 'Ekstrakcja encji' },
  { key: 'embedding',     label: 'Budowanie indeksu RAG' },
  { key: 'summarizing',   label: 'Generowanie podsumowania' },
]

export default function UploadPage() {
  const [dragging, setDragging] = useState(false)
  const [jsonError, setJsonError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)
  const { process, cancel } = useProcessingPipeline()
  const { session, clearSession, loadTranscript } = useSessionStore()
  const navigate = useNavigate()

  const isProcessing =
    session !== null &&
    session.processing.step !== 'idle' &&
    session.processing.step !== 'done' &&
    session.processing.step !== 'error'

  const handleFile = useCallback(
    (file: File) => {
      clearSession()
      process(file, 'whisper')
    },
    [process, clearSession],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const handleJsonFile = useCallback(
    (file: File) => {
      setJsonError(null)
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const raw = JSON.parse(e.target?.result as string)
          // Accept: array of segments, or object with .segments field
          const segsRaw: any[] = Array.isArray(raw)
            ? raw
            : Array.isArray(raw.segments)
            ? raw.segments
            : null
          if (!segsRaw) throw new Error('Nie znaleziono tablicy segmentów w pliku JSON.')
          const segments: Segment[] = segsRaw.map((s: any, i: number) => ({
            id: i,
            start: Number(s.start ?? 0),
            end: Number(s.end ?? 0),
            text: String(s.text ?? ''),
            text_pl: s.text_pl ? String(s.text_pl) : undefined,
            speaker: String(s.speaker ?? 'GŁOS_01'),
            language: s.language ? String(s.language) : undefined,
          }))
          const detectedLanguage = raw.detected_language ?? raw.language ?? 'auto'
          loadTranscript(file.name.replace(/\.json$/, ''), segments, detectedLanguage)
          navigate('/transcript')
        } catch (err: any) {
          setJsonError(err?.message ?? 'Błąd parsowania pliku JSON.')
        }
      }
      reader.readAsText(file)
    },
    [loadTranscript, navigate],
  )

  const currentStepIdx = STEPS.findIndex(
    (s) => s.key === session?.processing.step,
  )

  return (
    <div className="flex flex-col items-center justify-center min-h-full px-6 py-12">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-brand-50 rounded-2xl mb-4">
            <Mic className="w-8 h-8 text-brand-600" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">
            Pandaro
          </h1>
          <p className="text-slate-500 text-lg">
            Transkrypcja, diaryzacja i analiza AI nagrań audio
          </p>
        </div>

        {/* Model badge */}
        {!isProcessing && (
          <div className="mb-6 flex items-center gap-2 px-4 py-2.5 bg-brand-50 border border-brand-200 rounded-xl">
            <Mic className="w-4 h-4 text-brand-600 shrink-0" />
            <span className="text-sm font-medium text-brand-700">Whisper large-v3</span>
            <span className="text-xs text-brand-500 ml-1">— szybki, wielojęzyczny, diaryzacja SpeechBrain</span>
          </div>
        )}

        {/* Drop Zone */}
        {!isProcessing && (
          <div
            className={[
              'border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors',
              dragging
                ? 'border-brand-400 bg-brand-50'
                : 'border-slate-200 hover:border-brand-300 hover:bg-slate-50',
            ].join(' ')}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <Upload className="w-10 h-10 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-700 font-medium mb-1">
              Przeciągnij plik tutaj lub kliknij, aby wybrać
            </p>
            <p className="text-sm text-slate-400">
              MP3, MP4, M4A, WAV
            </p>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              className="hidden"
              onChange={onInputChange}
            />
          </div>
        )}

        {/* Load from JSON */}
        {!isProcessing && (
          <div className="mt-4">
            <button
              onClick={() => jsonInputRef.current?.click()}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50 hover:border-brand-300 transition-colors"
            >
              <FileJson className="w-4 h-4 text-slate-400" />
              Załaduj transkrypcję z pliku JSON
            </button>
            <input
              ref={jsonInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleJsonFile(f)
                e.target.value = ''
              }}
            />
            {jsonError && (
              <p className="mt-2 text-xs text-red-600 text-center">{jsonError}</p>
            )}
          </div>
        )}

        {/* Processing Pipeline */}
        {session && isProcessing && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-6">
              <Loader2 className="w-5 h-5 text-brand-600 animate-spin shrink-0" />
              <div>
                <p className="font-medium text-slate-800">{session.fileName}</p>
                <p className="text-sm text-slate-500">{session.processing.message}</p>
              </div>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-slate-100 rounded-full h-2 mb-6">
              <div
                className="bg-brand-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${session.processing.progress}%` }}
              />
            </div>

            {/* Steps */}
            <ul className="space-y-2">
              {STEPS.map((s, idx) => {
                const isDone = currentStepIdx > idx
                const isActive = s.key === session.processing.step
                return (
                  <li key={s.key} className="flex items-center gap-2.5 text-sm">
                    <span
                      className={[
                        'w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-xs font-bold',
                        isDone
                          ? 'bg-green-100 text-green-700'
                          : isActive
                          ? 'bg-brand-100 text-brand-700'
                          : 'bg-slate-100 text-slate-400',
                      ].join(' ')}
                    >
                      {isDone ? '✓' : idx + 1}
                    </span>
                    <span
                      className={
                        isDone
                          ? 'text-slate-400 line-through'
                          : isActive
                          ? 'text-brand-700 font-medium'
                          : 'text-slate-400'
                      }
                    >
                      {s.label}
                    </span>
                    {isActive && (
                      <Loader2 className="w-3.5 h-3.5 text-brand-400 animate-spin ml-auto" />
                    )}
                  </li>
                )
              })}
            </ul>
            <button
              onClick={cancel}
              className="mt-4 w-full text-sm text-slate-500 hover:text-red-600 underline hover:no-underline"
            >
              Anuluj
            </button>
          </div>
        )}

        {/* Error */}
        {session?.processing.step === 'error' && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Wystąpił błąd</p>
              <p className="text-sm text-red-600">{session.processing.error}</p>
              <button
                onClick={() => clearSession()}
                className="mt-2 text-sm text-red-700 underline hover:no-underline"
              >
                Spróbuj ponownie
              </button>
            </div>
          </div>
        )}

        {/* Supported formats note */}
        {!isProcessing && (
          <p className="text-center text-xs text-slate-400 mt-6">
            Transkrypcja odbywa się na serwerze (GPU). Do analizy LLM wymagany jest Ollama.
          </p>
        )}
      </div>
    </div>
  )
}
