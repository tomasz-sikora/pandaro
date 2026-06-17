import { api } from "../api";
import type { Analysis } from "../types";

interface Props {
  sessionId: string | null;
  analysis: Analysis | null;
  onClear: () => void;
  busy: boolean;
}

export function Toolbar({ sessionId, analysis, onClear, busy }: Props) {
  const conf = analysis?.confidence.mean_word_confidence;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo">🐼</span> Pandaro
        <span className="tagline">analiza nagrań — wszystko lokalnie i efemerycznie</span>
      </div>
      <div className="actions">
        {conf != null && (
          <span className="conf-badge" title="Średnia pewność rozpoznania">
            Pewność: {Math.round(conf * 100)}%
          </span>
        )}
        {sessionId && (
          <>
            <a href={api.exportUrl(sessionId, "pandaro")} download="analiza.pandaro">
              <button disabled={busy}>Eksport .pandaro</button>
            </a>
            <a href={api.exportUrl(sessionId, "srt")} download="napisy.srt">
              <button disabled={busy}>SRT</button>
            </a>
            <a href={api.exportUrl(sessionId, "md")} download="raport.md">
              <button disabled={busy}>Raport MD</button>
            </a>
          </>
        )}
        {(sessionId || analysis) && (
          <button className="danger" onClick={onClear} disabled={busy}>
            Wyczyść
          </button>
        )}
      </div>
    </header>
  );
}
