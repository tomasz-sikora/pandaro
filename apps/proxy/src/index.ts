import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'
import { createHash } from 'node:crypto'

const app = new Hono()

const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://host.docker.internal:11434'

// ── LRU cache ─────────────────────────────────────────────────────────────────
const CACHE_MAX = 10

/** Paths eligible for caching (never cache embed endpoints). */
const CACHEABLE_PATHS = new Set(['/api/generate', '/api/chat'])

class LRUCache<V> {
  private data = new Map<string, V>()
  constructor(private maxSize: number) {}

  get(key: string): V | undefined {
    if (!this.data.has(key)) return undefined
    const val = this.data.get(key)!
    this.data.delete(key)
    this.data.set(key, val)
    return val
  }

  set(key: string, value: V): void {
    if (this.data.has(key)) this.data.delete(key)
    else if (this.data.size >= this.maxSize)
      this.data.delete(this.data.keys().next().value as string)
    this.data.set(key, value)
  }

  info() {
    return { entries: this.data.size, maxsize: this.maxSize }
  }
}

interface CachedResponse {
  /** Aggregated text content (message.content for chat, response for generate). */
  text: string
  /** Original endpoint type */
  kind: 'chat' | 'generate'
  /** Model name, forwarded in fake stream */
  model: string
}

const ollamaCache = new LRUCache<CachedResponse>(CACHE_MAX)

function cacheKey(path: string, body: string): string {
  return createHash('sha256').update(path + '\n' + body).digest('hex')
}

/** Emit a fake Ollama streaming response from cached text. */
function fakeStream(cached: CachedResponse): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  const now = new Date().toISOString()
  let chunks: string[]
  if (cached.kind === 'chat') {
    chunks = [
      JSON.stringify({ model: cached.model, created_at: now, message: { role: 'assistant', content: cached.text }, done: false }),
      JSON.stringify({ model: cached.model, created_at: now, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' }),
    ]
  } else {
    chunks = [
      JSON.stringify({ model: cached.model, created_at: now, response: cached.text, done: false }),
      JSON.stringify({ model: cached.model, created_at: now, response: '', done: true }),
    ]
  }
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c + '\n'))
      controller.close()
    },
  })
}

// ── Middleware ─────────────────────────────────────────────────────────────────
app.use('*', logger())
app.use(
  '*',
  cors({
    origin: '*',
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
    exposeHeaders: ['Content-Type'],
    maxAge: 86400,
  }),
)

app.get('/health', (c) => c.json({ status: 'ok', ollama: OLLAMA_URL }))
app.get('/cache/info', (c) => c.json({ ollama_proxy: ollamaCache.info() }))

// ── Proxy with optional caching ────────────────────────────────────────────────
app.all('/*', async (c) => {
  const path = c.req.path
  const search = c.req.raw.url.includes('?')
    ? '?' + c.req.raw.url.split('?').slice(1).join('?')
    : ''
  const url = `${OLLAMA_URL}${path}${search}`
  const method = c.req.method

  // Only consider caching eligible POST paths
  const mayCacheThis = method === 'POST' && CACHEABLE_PATHS.has(path)

  let bodyText: string | undefined
  let parsedBody: Record<string, unknown> | undefined
  let key: string | undefined

  if (mayCacheThis) {
    bodyText = await c.req.text()
    try {
      parsedBody = JSON.parse(bodyText)
      key = cacheKey(path, bodyText)
      const hit = ollamaCache.get(key)
      if (hit) {
        console.log(`[ollama-cache] HIT  ${key.slice(0, 12)}… path=${path}`)
        const isStreaming = parsedBody.stream !== false
        if (isStreaming) {
          return new Response(fakeStream(hit), {
            status: 200,
            headers: { 'Content-Type': 'application/x-ndjson', 'X-Cache': 'HIT' },
          })
        } else {
          // Non-streaming: return plain JSON-like response
          const payload = hit.kind === 'chat'
            ? { model: hit.model, message: { role: 'assistant', content: hit.text }, done: true }
            : { model: hit.model, response: hit.text, done: true }
          return c.json({ ...payload, 'x-cache': 'HIT' })
        }
      }
    } catch {
      // unparseable body — skip caching
      key = undefined
    }
  }

  try {
    const upstream = await fetch(url, {
      method,
      headers: (() => {
        const h = new Headers()
        c.req.raw.headers.forEach((v, k) => {
          if (!['host', 'connection'].includes(k.toLowerCase())) h.set(k, v)
        })
        return h
      })(),
      body: bodyText !== undefined
        ? bodyText
        : (['GET', 'HEAD'].includes(method) ? undefined : c.req.raw.body),
      // @ts-ignore
      duplex: 'half',
    })

    const responseHeaders = new Headers()
    upstream.headers.forEach((v, k) => {
      if (!['transfer-encoding', 'connection'].includes(k.toLowerCase())) {
        responseHeaders.set(k, v)
      }
    })

    // If eligible for caching, buffer the stream to extract text content
    if (key && parsedBody && upstream.status === 200 && upstream.body) {
      const isStreaming = parsedBody.stream !== false
      const kind: 'chat' | 'generate' = path === '/api/chat' ? 'chat' : 'generate'
      const model = (parsedBody.model as string) ?? ''
      const finalKey = key

      if (isStreaming) {
        // Tee the stream: one branch for the client, one for our aggregator
        const [clientStream, aggregateStream] = upstream.body.tee()

        // Aggregate in background (do not await — client gets real stream)
        ;(async () => {
          const reader = aggregateStream.getReader()
          const dec = new TextDecoder()
          let buf = ''
          let text = ''
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
                  const obj = JSON.parse(line)
                  if (kind === 'chat') text += obj.message?.content ?? ''
                  else text += obj.response ?? ''
                } catch { /* skip malformed line */ }
              }
            }
            if (text) {
              ollamaCache.set(finalKey, { text, kind, model })
              console.log(`[ollama-cache] STORE ${finalKey.slice(0, 12)}… path=${path}`)
            }
          } catch { /* ignore aggregation errors */ } finally {
            reader.releaseLock()
          }
        })()

        return new Response(clientStream, { status: upstream.status, headers: responseHeaders })
      } else {
        // Non-streaming: buffer fully, cache, return
        const json = await upstream.json() as Record<string, unknown>
        const text: string = kind === 'chat'
          ? ((json.message as Record<string, unknown>)?.content as string) ?? ''
          : (json.response as string) ?? ''
        if (text) {
          ollamaCache.set(finalKey, { text, kind, model })
          console.log(`[ollama-cache] STORE ${finalKey.slice(0, 12)}… path=${path}`)
        }
        return c.json(json)
      }
    }

    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    })
  } catch (err: any) {
    return c.json({ error: `Proxy error: ${err?.message ?? 'unknown'}` }, 502)
  }
})

export default app
