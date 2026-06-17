import { formatTime } from "../text";
import type { Analysis } from "../types";
import { speakerColor } from "./TranscriptView";

// Panele analityczne: rozmówcy, słowa kluczowe, encje, cechy akustyczne/OSINT.
export function Dashboards({ analysis }: { analysis: Analysis }) {
  const a = analysis;
  return (
    <div className="dashboards">
      <div className="card">
        <h2>Rozmówcy</h2>
        {a.speakers.length ? (
          <ul className="speakers">
            {a.speakers.map((s) => (
              <li key={s.speaker}>
                <span className="dot" style={{ background: speakerColor(s.speaker) }} />
                <strong>{s.name || s.speaker}</strong>
                <span className="muted">
                  {" "}
                  {s.gender ?? "?"}, ~{s.age ?? "?"} lat · {formatTime(s.total_speech_s)} ·{" "}
                  {s.dominant_emotion ?? "?"}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">Brak danych o rozmówcach.</p>
        )}
      </div>

      <div className="card">
        <h2>Podsumowanie</h2>
        {a.summary.overall ? (
          <p className="summary">{a.summary.overall}</p>
        ) : (
          <p className="muted">Brak podsumowania.</p>
        )}
      </div>

      <div className="card">
        <h2>Słowa kluczowe</h2>
        <div className="cloud">
          {a.keywords.map((k) => (
            <span
              key={k.term}
              className="kw"
              style={{ fontSize: `${0.8 + k.score * 1.1}rem` }}
            >
              {k.term}
            </span>
          ))}
          {!a.keywords.length && <p className="muted">Brak.</p>}
        </div>
      </div>

      <div className="card">
        <h2>Encje</h2>
        <ul className="entities">
          {a.entities.slice(0, 30).map((e, i) => (
            <li key={i}>
              <span className={`tag tag-${e.type}`}>{e.type}</span> {e.text}
            </li>
          ))}
          {!a.entities.length && <p className="muted">Brak.</p>}
        </ul>
      </div>

      <div className="card">
        <h2>Cechy akustyczne / OSINT</h2>
        <table className="osint">
          <tbody>
            <tr><td>SNR</td><td>{fmt(a.acoustics.snr_db, "dB")}</td></tr>
            <tr><td>Szum tła</td><td>{fmt(a.acoustics.noise_floor_db, "dB")}</td></tr>
            <tr><td>Wysokość głosu (F0)</td><td>{fmt(a.acoustics.mean_pitch_hz, "Hz")}</td></tr>
            <tr><td>Tempo mowy</td><td>{fmt(a.acoustics.speech_rate_wps, "sł/s")}</td></tr>
            <tr><td>Cisza</td><td>{fmtPct(a.acoustics.silence_ratio)}</td></tr>
            <tr><td>Nakładanie się</td><td>{fmtPct(a.acoustics.overlap_ratio)}</td></tr>
            <tr><td>Tło</td><td>{a.acoustics.background_tags.join(", ") || "—"}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmt(v: number | null, unit: string): string {
  return v == null ? "—" : `${v} ${unit}`;
}
function fmtPct(v: number | null): string {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}
