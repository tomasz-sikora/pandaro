/// <reference lib="webworker" />
import { pipeline, env } from '@huggingface/transformers'

// Use browser cache for downloaded models
env.allowLocalModels = false
env.useBrowserCache = true

type InMsg =
  | { type: 'transcribe'; audioData: Float32Array; language?: string; modelId?: string }

type OutMsg =
  | { type: 'progress'; stage: string; progress: number; message: string }
  | { type: 'result'; chunks: Array<{ text: string; timestamp: [number, number] }>; detectedLanguage: string }
  | { type: 'error'; message: string }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let transcriber: any = null
let loadedModelId = ''

self.onmessage = async (event: MessageEvent<InMsg>) => {
  const msg = event.data

  if (msg.type === 'transcribe') {
    try {
      const modelId = msg.modelId ?? 'Xenova/whisper-small'

      if (!transcriber || loadedModelId !== modelId) {
        loadedModelId = modelId
        post({ type: 'progress', stage: 'loading_model', progress: 0, message: `Ładowanie modelu ${modelId}…` })

        transcriber = await pipeline('automatic-speech-recognition', modelId, {
          dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } as any,
          progress_callback: (p: any) => {
            if (p.status === 'progress' || p.status === 'downloading') {
              post({
                type: 'progress',
                stage: 'loading_model',
                progress: Math.round(p.progress ?? 0),
                message: `Pobieranie: ${p.file ?? ''} (${Math.round(p.progress ?? 0)}%)`,
              })
            }
          },
        } as any)
      }

      post({ type: 'progress', stage: 'transcribing', progress: 10, message: 'Transkrypcja w toku…' })

      const result = await (transcriber as any)(msg.audioData, {
        language: msg.language && msg.language !== 'auto' ? msg.language : null,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
        callback_function: (_: any) => {
          post({ type: 'progress', stage: 'transcribing', progress: 60, message: 'Transkrypcja w toku…' })
        },
      })

      const chunks: Array<{ text: string; timestamp: [number, number] }> =
        (result.chunks ?? []).map((c: any) => ({
          text: c.text as string,
          timestamp: c.timestamp as [number, number],
        }))

      post({
        type: 'result',
        chunks,
        detectedLanguage: (result as any).language ?? 'unknown',
      })
    } catch (err: any) {
      post({ type: 'error', message: err?.message ?? 'Nieznany błąd transkrypcji' })
    }
  }
}

function post(msg: OutMsg) {
  self.postMessage(msg)
}
