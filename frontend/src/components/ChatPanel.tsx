import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { runPython } from "../pyodide";
import type { RagIndex, SearchHit } from "../rag";
import { formatTime } from "../text";
import type { Analysis, RagChunk } from "../types";
import { speakerColor } from "./TranscriptView";

interface Props { index: RagIndex; analysis: Analysis; }

interface Source { chunk: RagChunk; score: number; }
interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  thinking?: boolean;
}

// ── Minimal markdown renderer (no deps) ─────────────────────────────────────
function mdToHtml(text: string): string {
  let h = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    // code blocks
    .replace(/```[\s\S]*?```/g, (m) => {
      const code = m.slice(3, -3).replace(/^[a-z]*\n/, "");
      return `<pre><code>${code}</code></pre>`;
    })
    // inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // bullet lists (lines starting with - • *)
    .replace(/^[•\-\*] (.+)/gm, "<li>$1</li>")
    // numbered lists
    .replace(/^\d+\. (.+)/gm, "<li>$1</li>")
    // wrap consecutive li in ul
    .replace(/(<li>[\s\S]*?<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    // headings
    .replace(/^## (.+)/gm, "<h3>$1</h3>")
    .replace(/^# (.+)/gm,  "<h2>$1</h2>")
    // paragraphs (double newline)
    .replace(/\n{2,}/g, "</p><p>")
    // single newline
    .replace(/\n/g, "<br>");
  return `<p>${h}</p>`;
}

// ── Transcript preview side panel ────────────────────────────────────────────
function TranscriptPreview({
  analysis,
  anchorTime,
  onClose,
  onSeek,
}: {
  analysis: Analysis;
  anchorTime: number;
  onClose: () => void;
  onSeek?: (t: number) => void;
}) {
  const segments = analysis.transcript.segments;
  const anchorRef = useRef<HTMLDivElement>(null);
  const anchorIdx = segments.reduce((best, seg, idx) => {
    return Math.abs(seg.start - anchorTime) < Math.abs(segments[best].start - anchorTime)
      ? idx
      : best;
  }, 0);

  useEffect(() => {
    anchorRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [anchorTime]);

  const visible = segments.slice(Math.max(0, anchorIdx - 8), Math.min(segments.length, anchorIdx + 8));

  return (
    <div className="chat-preview">
      <div className="chat-preview-header">
        <span className="hint">⏱ transkrypt @ {formatTime(anchorTime)}</span>
        <button className="ghost icon" onClick={onClose} title="Zamknij">✕</button>
      </div>
      <div className="chat-preview-segs">
        {visible.map((seg) => {
          const isAnchor = seg.id === segments[anchorIdx].id;
          const color = speakerColor(seg.speaker);
          return (
            <div
              key={seg.id}
              ref={isAnchor ? anchorRef : undefined}
              className={`preview-seg ${isAnchor ? "anchor" : ""}`}
              onClick={() => onSeek?.(seg.start)}
              style={{ cursor: onSeek ? "pointer" : "default" }}
            >
              <div style={{ display: "flex", gap: 6, marginBottom: 2 }}>
                <span className="dim" style={{ fontSize: 11, fontVariantNumeric: "tabular-nums", minWidth: 32 }}>
                  {formatTime(seg.start)}
                </span>
                {seg.speaker && (
                  <span style={{ color, fontSize: 11, fontWeight: 700 }}>{seg.speaker}</span>
                )}
              </div>
              <span style={{ fontSize: 12, lineHeight: 1.45 }}>{seg.text}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export function ChatPanel({ index, analysis }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [previewTime, setPreviewTime] = useState<number | null>(null);
  const [code, setCode] = useState(
    "# Dostępne: segments (lista), speakers (lista)\nprint('Segmentów:', len(segments))"
  );
  const [codeOut, setCodeOut] = useState("");
  const [showCode, setShowCode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollDown = useCallback(() => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 40);
  }, []);

  const send = useCallback(async () => {
    if (!input.trim() || busy) return;
    const question = input.trim();
    setInput("");
    setBusy(true);

    const uid = crypto.randomUUID();
    const aid = crypto.randomUUID();

    const userMsg: Msg = { id: uid, role: "user", content: question };
    setMessages((prev) => [...prev, userMsg, { id: aid, role: "assistant", content: "", thinking: false }]);
    scrollDown();

    // RAG retrieval
    let embedding: number[] | null = null;
    try { embedding = (await api.embed([question]))[0] ?? null; } catch { /* noop */ }
    const hits: SearchHit[] = index.search(question, embedding, 6);
    const context = hits.map((h) => ({
      text: h.chunk.text, speaker: h.chunk.speaker, start: h.chunk.start, end: h.chunk.end,
    }));
    const sources: Source[] = hits.map((h) => ({ chunk: h.chunk, score: h.score }));

    const history = messages
      .concat(userMsg)
      .map((m) => ({ role: m.role, content: m.content }));

    // Stream response
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await api.streamChat(
        history, context,
        (chunk) => {
          setMessages((prev) =>
            prev.map((m) => m.id === aid ? { ...m, content: m.content + chunk, sources, thinking: false } : m)
          );
          scrollDown();
        },
        ctrl.signal,
        (isThinking) => {
          setMessages((prev) =>
            prev.map((m) => m.id === aid ? { ...m, thinking: isThinking } : m)
          );
        }
      );
    } catch (e: any) {
      setMessages((prev) =>
        prev.map((m) => m.id === aid ? { ...m, content: `❌ Błąd: ${String(e)}`, sources, thinking: false } : m)
      );
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [input, busy, messages, index, scrollDown]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const exec = async () => {
    setCodeOut("Uruchamianie (Pyodide)…");
    const res = await runPython(code, {
      segments: analysis.transcript.segments.map((s) => ({ start: s.start, end: s.end, text: s.text, speaker: s.speaker })),
      speakers: analysis.speakers.map((s) => ({ ...s })),
    });
    setCodeOut(res.ok ? `${res.stdout}${res.result ?? ""}` : `Błąd: ${res.error}`);
  };

  return (
    <div className="chat-shell">
      {/* ── Main chat column ── */}
      <div className="chat-main">
        <div className="chat-header">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <h2>Asystent RAG</h2>
              <span className="hint">
                {index.size
                  ? `${index.size} fragmentów w indeksie · gemma4:31b`
                  : "⚠ Indeks RAG nie jest gotowy — uruchom fazę RAG"}
              </span>
            </div>
            {messages.length > 0 && (
              <button className="ghost sm" onClick={() => setMessages([])}>🗑 Wyczyść</button>
            )}
          </div>
        </div>

        <div className="chat-messages">
          {!messages.length && (
            <div className="chat-empty">
              <div className="chat-empty-icon">🤖</div>
              <div style={{ fontWeight: 600, fontSize: 14 }}>Zadaj pytanie o nagranie</div>
              <div className="hint" style={{ maxWidth: 280 }}>
                Odpowiedzi zawierają cytaty z transkryptu. Kliknij źródło, żeby przejść do fragmentu.
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`msg-row ${msg.role}`}>
              {msg.role === "assistant" && (
                <div className="msg-avatar bot">🤖</div>
              )}
              <div className={`msg-bubble ${msg.role === "user" ? "user" : "bot"}`}>
                {msg.role === "user" ? (
                  <span>{msg.content}</span>
                ) : msg.thinking && !msg.content ? (
                  <span style={{ color: "var(--text-2)", fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
                    <span className="spinner" /> Rozważam…
                  </span>
                ) : msg.content ? (
                  <>
                    {msg.thinking && (
                      <span style={{ color: "var(--text-3)", fontSize: 11, display: "flex", gap: 4, alignItems: "center", marginBottom: 6 }}>
                        <span className="spinner" style={{ width: 10, height: 10 }} /> rozważam…
                      </span>
                    )}
                    <div dangerouslySetInnerHTML={{ __html: mdToHtml(msg.content) }} />
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="msg-sources">
                        <div className="sources-label">Źródła</div>
                        {msg.sources.map((src, i) => (
                          <button
                            key={i}
                            className="source-item"
                            onClick={() => setPreviewTime(src.chunk.start)}
                            title={`Otwórz fragment @ ${formatTime(src.chunk.start)}`}
                          >
                            <span className="source-num">[{i + 1}]</span>
                            <span className="source-time">{formatTime(src.chunk.start)}</span>
                            <span className="source-text">{src.chunk.text.slice(0, 100)}{src.chunk.text.length > 100 ? "…" : ""}</span>
                            <span className="source-score">{Math.round(src.score * 100)}%</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                ) : !msg.thinking ? (
                  <span className="spinner" />
                ) : null}
              </div>
              {msg.role === "user" && (
                <div className="msg-avatar user">👤</div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-area">
          {/* Pyodide toggle */}
          <div style={{ marginBottom: 8 }}>
            <button className="ghost sm" onClick={() => setShowCode((s) => !s)}>
              {showCode ? "▲" : "▼"} Interpreter kodu (Pyodide)
            </button>
          </div>
          {showCode && (
            <div className="code-panel" style={{ marginBottom: 10 }}>
              <p className="hint">Zmienne: <code>segments</code>, <code>speakers</code></p>
              <textarea value={code} onChange={(e) => setCode(e.target.value)} rows={4} />
              <div className="row">
                <button onClick={exec}>▶ Uruchom</button>
                <button className="ghost sm" onClick={() => setCodeOut("")}>Wyczyść</button>
              </div>
              {codeOut && <pre className="code-out">{codeOut}</pre>}
            </div>
          )}
          <div className="chat-input-row">
            <input
              className="chat-input"
              type="text"
              placeholder="Zadaj pytanie o nagranie…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
              disabled={busy}
            />
            {busy ? (
              <button className="chat-send" onClick={stop} title="Zatrzymaj">⏹</button>
            ) : (
              <button
                className="chat-send"
                onClick={send}
                disabled={!input.trim() || !index.size}
                title="Wyślij"
              >
                ➤
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Transcript preview panel ── */}
      {previewTime !== null && analysis.transcript.segments.length > 0 && (
        <TranscriptPreview
          analysis={analysis}
          anchorTime={previewTime}
          onClose={() => setPreviewTime(null)}
        />
      )}
    </div>
  );
}
