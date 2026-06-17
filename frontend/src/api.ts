// Klient HTTP/WebSocket do backendu Pandaro.

import type { Analysis, PhaseState } from "./types";

const BASE = "";

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export interface PublicConfig {
  llm_model: string;
  llm_model_fallback: string;
  embedding_model: string;
  default_language: string;
  confidence_threshold: number;
  phases: string[];
}

export const api = {
  async config(): Promise<PublicConfig> {
    return jsonOrThrow(await fetch(`${BASE}/api/config`));
  },

  async health(): Promise<any> {
    return jsonOrThrow(await fetch(`${BASE}/api/health`));
  },

  async createSession(file: File, presetJson: string): Promise<{ session_id: string; analysis: Analysis }> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("preset", presetJson);
    return jsonOrThrow(await fetch(`${BASE}/api/sessions`, { method: "POST", body: fd }));
  },

  async run(sid: string): Promise<void> {
    await jsonOrThrow(await fetch(`${BASE}/api/sessions/${sid}/run`, { method: "POST" }));
  },

  async rerunPhase(sid: string, phase: string): Promise<void> {
    await jsonOrThrow(
      await fetch(`${BASE}/api/sessions/${sid}/phases/${phase}`, { method: "POST" })
    );
  },

  async getSession(sid: string): Promise<Analysis> {
    return jsonOrThrow(await fetch(`${BASE}/api/sessions/${sid}`));
  },

  async clearSession(sid: string): Promise<void> {
    await fetch(`${BASE}/api/sessions/${sid}`, { method: "DELETE" }).catch(() => {});
  },

  async cancelSession(sid: string): Promise<void> {
    await fetch(`${BASE}/api/sessions/${sid}/cancel`, { method: "POST" }).catch(() => {});
  },

  async embed(texts: string[]): Promise<number[][]> {
    const body = JSON.stringify({ texts });
    const res = await fetch(`${BASE}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const data = await jsonOrThrow(res);
    return data.embeddings ?? [];
  },

  async chat(
    messages: { role: string; content: string }[],
    context: { text: string; speaker: string | null; start: number; end: number }[]
  ): Promise<string> {
    const res = await fetch(`${BASE}/api/llm/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, context }),
    });
    const data = await jsonOrThrow(res);
    return data.answer ?? "";
  },

  /** Streaming chat — yields text chunks via SSE. Calls onThinking(true/false) for thinking models. */
  streamChat(
    messages: { role: string; content: string }[],
    context: { text: string; speaker: string | null; start: number; end: number }[],
    onChunk: (chunk: string) => void,
    signal: AbortSignal,
    onThinking?: (thinking: boolean) => void,
  ): Promise<void> {
    return new Promise(async (resolve, reject) => {
      try {
        const res = await fetch(`${BASE}/api/llm/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages, context }),
          signal,
        });
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6);
            if (raw === "[DONE]") { resolve(); return; }
            const parsed = JSON.parse(raw);
            if (parsed.error) throw new Error(parsed.error);
            if (parsed.thinking !== undefined) onThinking?.(parsed.thinking);
            else if (parsed.chunk) onChunk(parsed.chunk);
          }
        }
        resolve();
      } catch (e: any) {
        if (e?.name === "AbortError") resolve();
        else reject(e);
      }
    });
  },

  async importBundle(file: File): Promise<{ session_id: string; analysis: Analysis }> {
    const fd = new FormData();
    fd.append("file", file);
    return jsonOrThrow(await fetch(`${BASE}/api/import`, { method: "POST", body: fd }));
  },

  exportUrl(sid: string, fmt: string): string {
    return `${BASE}/api/sessions/${sid}/export?fmt=${fmt}`;
  },

  openProgressSocket(sid: string, onState: (s: PhaseState) => void): WebSocket {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/sessions/${sid}/ws`);
    ws.onmessage = (ev) => {
      try {
        onState(JSON.parse(ev.data));
      } catch {
        /* ignore */
      }
    };
    return ws;
  },
};
