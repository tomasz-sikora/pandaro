import { PHASE_LABELS, type Analysis } from "../types";

interface Props {
  analysis: Analysis;
  onRerun: (phase: string) => void;
  busy: boolean;
}

const STATUS_LABEL: Record<string, string> = {
  pending: "oczekuje",
  running: "w toku…",
  done: "gotowe",
  skipped: "pominięto",
  error: "błąd",
};

export function ProgressPanel({ analysis, onRerun, busy }: Props) {
  const phases = Object.values(analysis.phases);
  return (
    <div className="card">
      <h2>Postęp analizy</h2>
      <ul className="phases">
        {phases.map((p) => (
          <li key={p.phase} className={`phase phase-${p.status}`}>
            <span className="phase-name">{PHASE_LABELS[p.phase] ?? p.phase}</span>
            <span className="phase-status">{STATUS_LABEL[p.status]}</span>
            {p.error && <span className="phase-error" title={p.error}>⚠</span>}
            <button
              className="link"
              disabled={busy}
              title="Uruchom ponownie ten etap"
              onClick={() => onRerun(p.phase)}
            >
              ⟳
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
