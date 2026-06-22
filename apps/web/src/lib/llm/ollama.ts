export interface OllamaConfig {
  baseUrl: string
  model: string
  embeddingModel: string
}

export interface OllamaMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

/** Stream a chat completion from Ollama, yielding text deltas. */
export async function* ollamaChat(
  cfg: OllamaConfig,
  messages: OllamaMessage[],
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const res = await fetch(`${cfg.baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: cfg.model, messages, stream: true }),
    signal,
  })

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`Ollama ${res.status}: ${body}`)
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
        if (!line.trim()) continue
        try {
          const data = JSON.parse(line)
          if (data.message?.content) yield data.message.content as string
        } catch {
          // skip malformed line
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Non-streaming completion — uses stream:false for a single JSON response.
 * Much lighter on memory than accumulating streaming chunks.
 * @param timeoutMs  Abort after this many ms (default: 3 minutes)
 */
export async function ollamaComplete(
  cfg: OllamaConfig,
  prompt: string,
  signal?: AbortSignal,
  timeoutMs = 180_000,
): Promise<string> {
  // Combine caller signal with an internal timeout signal
  const timeoutCtrl = new AbortController()
  const timer = setTimeout(() => timeoutCtrl.abort(), timeoutMs)

  const combined =
    signal
      ? AbortSignal.any
        ? AbortSignal.any([signal, timeoutCtrl.signal])
        : timeoutCtrl.signal          // fallback: at least honour timeout
      : timeoutCtrl.signal

  try {
    const res = await fetch(`${cfg.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: cfg.model,
        messages: [{ role: 'user', content: prompt }],
        stream: false,
      }),
      signal: combined,
    })

    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`Ollama ${res.status}: ${body}`)
    }

    const data = await res.json()
    return (data.message?.content as string) ?? ''
  } finally {
    clearTimeout(timer)
  }
}

/** Generate embeddings via Ollama /api/embed */
export async function ollamaEmbed(
  cfg: OllamaConfig,
  texts: string[],
  batchSize = 20,
): Promise<number[][]> {
  // Send in batches to avoid request-size / timeout issues with large transcripts
  const results: number[][] = []
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const res = await fetch(`${cfg.baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: cfg.embeddingModel, input: batch }),
    })
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`Ollama embed ${res.status}: ${body}`)
    }
    const data = await res.json()
    results.push(...(data.embeddings as number[][]))
  }
  return results
}

/** List available Ollama models */
export async function ollamaListModels(baseUrl: string): Promise<string[]> {
  const res = await fetch(`${baseUrl}/api/tags`)
  if (!res.ok) return []
  const data = await res.json()
  return (data.models ?? []).map((m: { name: string }) => m.name)
}
