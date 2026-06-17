import { PHASE_LABELS, type Analysis } from "../types";

interface Props {
  analysis: Analysis;
  onRerun: (phase: string) => void;
  onCancel: () => void;
  busy: boolean;
}

const STATUS_ICON: Record<string, string> = {
  pending:  "○",
  running:  "⟳",
  done:     "✓",
  skipped:  "—",
  error:    "✕",
};

const PHASE_TIPS: Record<string, string> = {
  ingest:         "Wczytywanie i konwersja audio do 16 kHz mono",
  vad:            "Detekcja fragmentów z mową (Silero VAD)",
  asr:            "Transkrypcja mowy na tekst (Whisper large-v3)",
  align:          "Wyrównanie słów do osi czasu (wav2vec2)",
  diarize:        "Przypisanie rozmówców do segmentów (pyannote)",
  merge:          "Łączenie transkryptu z diaryzacją",
  speaker_id:     "Identyfikacja rozmówców wg podpowiedzi",
  paralinguistics:"Analiza głosu — wiek, płeć, emocje",
  acoustics:      "Cechy akustyczne: SNR, F0, tempo, tło",
  translate:      "Tłumaczenie segmentów na polski",
  keywords:       "Ekstrakcja słów kluczowych i encji (LLM)",
  summarize:      "Hierarchiczne podsumowanie nagrania (LLM)",
  rag:            "Budowa hybrydowego indeksu RAG",
  report:         "Generowanie raportu końcowego",
};

function elapsed(st: number | null | undefined, end: number | null | undefined): string {
  if (!st) return "";
  const t = ((end ?? Date.now() / 1000) - st);
  return t < 60 ? `${t.toFixed(1)}s` : `${Math.floor(t / 60)}m ${Math.round(t % 60)}s`;
}

export function ProgressPanel({ analysis, onRerun, onCancel, busy }: Props) {
  const phases = Object.values(analysis.phases);
  const done = phases.filter(p => p.status === "done").length;
  const total = phases.filter(p => p.status !== "skipped").length;
  const hasRunning = phases.some(p => p.status === "running");
  const errors = phases.filter(p => p.status === "error");

  return (
    <div className="progress-layout">
      <div className="progress-header">
        <div>
          <h2 style={{ margin: 0 }}>Postęp analizy</h2>
          <div className="progress-summary">
            {done}/{total} faz ukończonych
            {errors.length > 0 && (
              <span style={{ color: "var(--danger)", marginLeft: 8 }}>
                · {errors.length} {errors.length === 1 ? "błąd" : "błędy"}
              </span>
            )}
          </div>
        </div>
        <div className="progress-controls">
          {hasRunning && (
            <button className="danger" onClick={onCancel} disabled={busy}>
              ⏹ Przerwij
            </button>
          )}
        </div>
      </div>

      <div className="card">
        <div className="phase-list">
          {phases.map((p) => {
            const pct = p.status === "done" ? 100
              : p.status === "running" ? Math.round((p.progress ?? 0) * 100)
              : 0;
            return (
              <div key={p.phase} className={`phase-item ${p.status}`} title={PHASE_TIPS[p.phase]}>
                <div className="phase-icon">
                  {p.status === "running"
                    ? <span className="spinner" />
                    : STATUS_ICON[p.status] ?? "○"}
                </div>
                <div className="phase-body">
                  <div className="phase-name">{PHASE_LABELS[p.phase] ?? p.phase}</div>
                  {p.error && (
                    <div className="phase-detail err" title={p.error}>
                      {p.error.slice(0, 200)}
                    </div>
                  )}
                  {!p.error && p.message && (
                    <div className="phase-detail">{p.message}</div>
                  )}
                  {!p.error && !p.message && p.status === "running" && (
                    <div className="phase-detail">Trwa przetwarzanie…</div>
                  )}
                  {p.status === "done" && p.started_at && (
                    <div className="phase-detail">
                      Czas: {elapsed(p.started_at, p.ended_at)}
                    </div>
                  )}
                </div>
                <div className="phase-bar-wrap">
                  <div
                    className="phase-bar"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="phase-actions">
                  {(p.status === "done" || p.status === "error") && (
                    <button
                      className="ghost icon"
                      title="Uruchom ponownie"
                      onClick={() => onRerun(p.phase)}
                    >
                      ↺
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
