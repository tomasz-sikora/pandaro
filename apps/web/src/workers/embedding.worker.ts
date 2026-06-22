/// <reference lib="webworker" />
import { pipeline, env } from '@huggingface/transformers'

env.allowLocalModels = false
env.useBrowserCache = false
env.allowRemoteModels = true

type InMsg = { type: 'embed'; texts: string[]; modelId?: string }
type OutMsg =
  | { type: 'progress'; progress: number; message: string }
  | { type: 'result'; embeddings: number[][] }
  | { type: 'error'; message: string }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let extractor: any = null
let loadedModelId = ''

self.onmessage = async (event: MessageEvent<InMsg>) => {
  const msg = event.data
  if (msg.type !== 'embed') return

  try {
    const modelId = msg.modelId ?? 'Xenova/all-MiniLM-L6-v2'

    if (!extractor || loadedModelId !== modelId) {
      loadedModelId = modelId
      post({ type: 'progress', progress: 0, message: `Ładowanie modelu embeddingów…` })

      extractor = await pipeline('feature-extraction', modelId, {
        progress_callback: (p: any) => {
          if (p.status === 'progress' || p.status === 'downloading') {
            post({
              type: 'progress',
              progress: Math.round(p.progress ?? 0),
              message: `Pobieranie: ${p.file ?? ''} (${Math.round(p.progress ?? 0)}%)`,
            })
          }
        },
      } as any)
    }

    post({ type: 'progress', progress: 50, message: 'Generowanie embeddingów…' })

    const output = await (extractor as any)(msg.texts, {
      pooling: 'mean',
      normalize: true,
    })

    const embeddings: number[][] = output.tolist()
    post({ type: 'result', embeddings })
  } catch (err: any) {
    post({ type: 'error', message: err?.message ?? 'Błąd embeddingów' })
  }
}

function post(msg: OutMsg) {
  self.postMessage(msg)
}
