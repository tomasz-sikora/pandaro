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
