import { useState } from "react";
import { api } from "../api";
import { runPython } from "../pyodide";
import type { RagIndex } from "../rag";
import { formatTime } from "../text";
import type { Analysis } from "../types";

interface Props {
  index: RagIndex;
  analysis: Analysis;
}

interface Msg {
  role: "user" | "assistant";
  content: string;
}

// Czat agenta z RAG (cytaty z dowodami) + interpreter kodu Pyodide do
// dalszego przetwarzania wyników analizy w przeglądarce.
export function ChatPanel({ index, analysis }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [code, setCode] = useState(
    "# Dostępne zmienne: segments (lista), speakers (lista)\n" +
      "print('Liczba segmentów:', len(segments))"
  );
  const [codeOut, setCodeOut] = useState("");
  const [showCode, setShowCode] = useState(false);

  const ask = async () => {
    if (!input.trim()) return;
    const userMsg: Msg = { role: "user", content: input };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setBusy(true);

    let embedding: number[] | null = null;
    try {
      embedding = (await api.embed([userMsg.content]))[0] ?? null;
    } catch {
      embedding = null;
    }
    const hits = index.search(userMsg.content, embedding);
    const context = hits.map((h) => ({
      text: h.chunk.text,
      speaker: h.chunk.speaker,
      start: h.chunk.start,
      end: h.chunk.end,
    }));

    try {
      const answer = await api.chat(history, context);
      setMessages([...history, { role: "assistant", content: answer }]);
    } catch (e) {
      setMessages([
        ...history,
        { role: "assistant", content: `Błąd połączenia z modelem: ${String(e)}` },
      ]);
    }
    setBusy(false);
  };

  const exec = async () => {
    setCodeOut("Uruchamianie (Pyodide)…");
    const res = await runPython(code, {
      segments: analysis.transcript.segments.map((s) => ({
        start: s.start,
        end: s.end,
        text: s.text,
        speaker: s.speaker,
      })),
      speakers: analysis.speakers.map((s) => ({ ...s })),
    });
    setCodeOut(res.ok ? `${res.stdout}${res.result ?? ""}` : `Błąd: ${res.error}`);
  };

  return (
    <div className="card chat">
      <h2>Asystent (RAG)</h2>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            {m.content}
          </div>
        ))}
        {!messages.length && (
          <p className="muted">
            Zadaj pytanie o nagranie. Odpowiedzi zawierają cytaty [ROZMÓWCA, czas] jako dowód.
          </p>
        )}
      </div>
      <div className="row">
        <input
          type="text"
          placeholder="np. O czym rozmawiano w sprawie umowy?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={busy}
        />
        <button onClick={ask} disabled={busy || !index.size}>
          {busy ? "…" : "Wyślij"}
        </button>
      </div>

      <button className="link" onClick={() => setShowCode((s) => !s)}>
        {showCode ? "▾" : "▸"} Interpreter kodu (Pyodide)
      </button>
      {showCode && (
        <div className="code-interp">
          <textarea value={code} onChange={(e) => setCode(e.target.value)} rows={5} />
          <button onClick={exec}>Uruchom</button>
          <pre className="code-out">{codeOut}</pre>
          <p className="hint">
            Czas ostatniego segmentu:{" "}
            {analysis.transcript.segments.length
              ? formatTime(
                  analysis.transcript.segments[analysis.transcript.segments.length - 1].end
                )
              : "—"}
          </p>
        </div>
      )}
    </div>
  );
}
