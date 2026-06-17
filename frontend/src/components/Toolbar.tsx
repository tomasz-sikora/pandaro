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
  const confClass = conf == null ? "" : conf >= 0.8 ? " good" : conf >= 0.6 ? " warn" : "";

  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo">🐼</span>
        <span>Pandaro</span>
        <span className="tagline">analiza nagrań · lokalnie · efemerycznie</span>
      </div>
      <div className="topbar-spacer" />
      <div className="topbar-actions">
        {conf != null && (
          <span className={`conf-badge${confClass}`} title="Średnia pewność rozpoznania mowy">
            ASR: {Math.round(conf * 100)}%
          </span>
        )}
        {analysis?.media_filename && (
          <span className="conf-badge" title="Plik nagrania">
            📁 {analysis.media_filename}
          </span>
        )}
        {sessionId && (
          <>
            <a href={api.exportUrl(sessionId, "pandaro")} download="analiza.pandaro" title="Eksportuj pełną analizę">
              <button disabled={busy}>💾 .pandaro</button>
            </a>
            <a href={api.exportUrl(sessionId, "srt")} download="napisy.srt" title="Eksportuj napisy">
              <button disabled={busy}>SRT</button>
            </a>
            <a href={api.exportUrl(sessionId, "md")} download="raport.md" title="Eksportuj raport Markdown">
              <button disabled={busy}>MD</button>
            </a>
          </>
        )}
        {(sessionId || analysis) && (
          <button className="danger" onClick={onClear} disabled={busy} title="Wyczyść sesję (ephemeryczność)">
            🗑 Wyczyść
          </button>
        )}
      </div>
    </header>
  );
}
