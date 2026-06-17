import { useRef, useState } from "react";
import { api } from "../api";
import { runPython } from "../pyodide";
import type { RagIndex } from "../rag";
import { formatTime } from "../text";
import type { Analysis } from "../types";

interface Props {
  index: RagIndex;
  analysis: Analysis;
}
interface Msg { role: "user" | "assistant"; content: string; }

export function ChatPanel({ index, analysis }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [code, setCode] = useState(
    "# Dostępne: segments (lista), speakers (lista)\nprint('Segmentów:', len(segments))"
  );
  const [codeOut, setCodeOut] = useState("");
  const [showCode, setShowCode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollDown = () =>
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

  const ask = async () => {
    if (!input.trim()) return;
    const userMsg: Msg = { role: "user", content: input };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setBusy(true);
    scrollDown();

    let embedding: number[] | null = null;
    try { embedding = (await api.embed([userMsg.content]))[0] ?? null; } catch { /* noop */ }
    const hits = index.search(userMsg.content, embedding);
    const context = hits.map((h) => ({ text: h.chunk.text, speaker: h.chunk.speaker, start: h.chunk.start, end: h.chunk.end }));

    try {
      const answer = await api.chat(history, context);
      setMessages([...history, { role: "assistant", content: answer }]);
    } catch (e) {
      setMessages([...history, { role: "assistant", content: `❌ Błąd: ${String(e)}` }]);
    }
    setBusy(false);
    scrollDown();
  };

  const exec = async () => {
    setCodeOut("Uruchamianie (Pyodide)…");
    const res = await runPython(code, {
      segments: analysis.transcript.segments.map((s) => ({ start: s.start, end: s.end, text: s.text, speaker: s.speaker })),
      speakers: analysis.speakers.map((s) => ({ ...s })),
    });
    setCodeOut(res.ok ? `${res.stdout}${res.result ?? ""}` : `Błąd: ${res.error}`);
  };

  return (
    <>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Asystent RAG</h2>
          {messages.length > 0 && (
            <button className="ghost" style={{ fontSize: 12 }} onClick={() => setMessages([])}>
              🗑 Wyczyść czat
            </button>
          )}
        </div>
        <p className="hint" style={{ marginBottom: 10 }}>
          Odpowiedzi cytują fragmenty [ROZMÓWCA, mm:ss]. Kliknij cytat w transkrypcie.
        </p>

        <div className="chat-messages">
          {!messages.length && (
            <div className="chat-empty">
              Zadaj pytanie o nagranie — asystent opiera się wyłącznie na transkrypcie.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg msg-${m.role}`}>{m.content}</div>
          ))}
          {busy && (
            <div className="msg msg-assistant">
              <span className="spinner" />
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-row">
          <input
            type="text"
            placeholder="np. O czym rozmawiano w sprawie umowy?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy && ask()}
            disabled={busy}
          />
          <button className="primary" onClick={ask} disabled={busy || !index.size}>
            Wyślij
          </button>
        </div>
        {!index.size && <p className="hint" style={{ marginTop: 8 }}>⚠ Indeks RAG nie jest gotowy.</p>}
      </div>

      {/* Interpreter kodu */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <h2 style={{ margin: 0 }}>Interpreter kodu (Pyodide)</h2>
          <button className="ghost" onClick={() => setShowCode((s) => !s)}>
            {showCode ? "▲ Zwiń" : "▼ Rozwiń"}
          </button>
        </div>
        {showCode && (
          <div className="code-section">
            <p className="hint">
              Python w przeglądarce (WASM). Zmienne: <code>segments</code>, <code>speakers</code>.
              Czas nagrania: {analysis.transcript.segments.length
                ? formatTime(analysis.transcript.segments[analysis.transcript.segments.length - 1]?.end ?? 0)
                : "—"}
            </p>
            <textarea value={code} onChange={(e) => setCode(e.target.value)} rows={6} />
            <div className="row">
              <button onClick={exec}>▶ Uruchom</button>
              <button className="ghost" onClick={() => setCodeOut("")}>Wyczyść wynik</button>
            </div>
            {codeOut && <pre className="code-out">{codeOut}</pre>}
          </div>
        )}
      </div>
    </>
  );
}
