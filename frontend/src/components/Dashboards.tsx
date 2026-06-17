import { formatTime } from "../text";
import type { Analysis } from "../types";
import { speakerColor } from "./TranscriptView";

interface Props {
  analysis: Analysis;
  onRerun: (phase: string) => void;
}

function fmt(v: number | null | undefined, unit: string): string {
  return v == null ? "—" : `${typeof v === "number" ? v.toFixed(1) : v} ${unit}`;
}
function fmtPct(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

export function Dashboards({ analysis: a, onRerun }: Props) {
  const maxSpeech = Math.max(...a.speakers.map((s) => s.total_speech_s), 1);

  return (
    <div className="analysis-grid">

      {/* Rozmówcy */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Rozmówcy</h2>
          {!a.speakers.length && (
            <button className="ghost" style={{ fontSize: 12 }} onClick={() => onRerun("paralinguistics")}>↺ Uruchom analizę głosu</button>
          )}
        </div>
        {a.speakers.length ? (
          <ul className="speakers-list">
            {a.speakers.map((s) => {
              const color = speakerColor(s.speaker);
              return (
                <li key={s.speaker} className="speaker-item">
                  <span className="spk-dot" style={{ background: color }} />
                  <div className="spk-info">
                    <div className="spk-name">{s.name || s.speaker}</div>
                    <div className="spk-meta">
                      {[
                        s.gender === "male" ? "mężczyzna" : s.gender === "female" ? "kobieta" : null,
                        s.age != null ? `~${s.age.toFixed(0)} lat` : null,
                        s.dominant_emotion,
                        formatTime(s.total_speech_s) + " mowy",
                      ].filter(Boolean).join(" · ")}
                    </div>
                    <div className="spk-bar-wrap">
                      <div
                        className="spk-bar"
                        style={{ background: color, width: `${(s.total_speech_s / maxSpeech) * 100}%` }}
                      />
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="empty">Brak profili rozmówców.</div>
        )}
      </div>

      {/* Pewność ASR */}
      <div className="card">
        <h2>Jakość transkrypcji</h2>
        <div className="stat-grid">
          <div className="stat-box">
            <div className="stat-label">Średnia pewność</div>
            <div className="stat-value">{Math.round(a.confidence.mean_word_confidence * 100)}%</div>
          </div>
          <div className="stat-box">
            <div className="stat-label">Niska pewność</div>
            <div className="stat-value">{Math.round(a.confidence.low_confidence_ratio * 100)}%</div>
          </div>
          <div className="stat-box">
            <div className="stat-label">Czas trwania</div>
            <div className="stat-value sm">{formatTime(a.media_duration)}</div>
          </div>
          <div className="stat-box">
            <div className="stat-label">Segmentów</div>
            <div className="stat-value sm">{a.transcript.segments.length}</div>
          </div>
        </div>
        {Object.keys(a.confidence.per_speaker).length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="hint" style={{ marginBottom: 6 }}>Pewność wg rozmówcy</div>
            {Object.entries(a.confidence.per_speaker).map(([spk, conf]) => (
              <div key={spk} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
                <span style={{ color: speakerColor(spk), fontSize: 12, minWidth: 80 }}>{spk}</span>
                <div style={{ flex: 1, height: 5, background: "var(--bg)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${conf * 100}%`, height: "100%", background: conf > 0.7 ? "var(--success)" : "var(--warning)", borderRadius: 3 }} />
                </div>
                <span className="hint">{Math.round(conf * 100)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Podsumowanie */}
      <div className="card analysis-full">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Podsumowanie</h2>
          <button className="ghost" style={{ fontSize: 12 }} onClick={() => onRerun("summarize")}>↺ Przebuduj</button>
        </div>
        {a.summary.overall ? (
          <>
            <p className="summary-text">{a.summary.overall}</p>
            {Object.entries(a.summary.per_speaker).map(([spk, txt]) => (
              <div key={spk} style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
                <div style={{ fontWeight: 600, color: speakerColor(spk), marginBottom: 4 }}>{spk}</div>
                <p className="summary-text">{txt}</p>
              </div>
            ))}
          </>
        ) : (
          <div className="empty">Brak podsumowania — uruchom fazę Podsumowanie.</div>
        )}
      </div>

      {/* Słowa kluczowe */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Słowa kluczowe</h2>
          <button className="ghost" style={{ fontSize: 12 }} onClick={() => onRerun("keywords")}>↺ Przebuduj</button>
        </div>
        {a.keywords.length ? (
          <div className="kw-cloud">
            {a.keywords.slice(0, 40).map((k) => (
              <span key={k.term} className="kw-tag" style={{ fontSize: `${0.78 + k.score * 0.6}rem` }}>
                {k.term}
              </span>
            ))}
          </div>
        ) : (
          <div className="empty">Brak słów kluczowych.</div>
        )}
      </div>

      {/* Encje */}
      <div className="card">
        <h2>Encje nazwane</h2>
        {a.entities.length ? (
          <div className="entities-list">
            {a.entities.slice(0, 40).map((e, i) => (
              <div key={i} className="entity-row">
                <span className={`entity-type ${e.type}`}>{e.type}</span>
                <span>{e.text}</span>
                {e.count > 1 && <span className="hint">×{e.count}</span>}
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">Brak encji.</div>
        )}
      </div>

      {/* Cechy akustyczne */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Cechy akustyczne / OSINT</h2>
          <button className="ghost" style={{ fontSize: 12 }} onClick={() => onRerun("acoustics")}>↺ Przebuduj</button>
        </div>
        <table className="osint-table">
          <tbody>
            <tr><td>SNR</td><td>{fmt(a.acoustics.snr_db, "dB")}</td></tr>
            <tr><td>Szum tła</td><td>{fmt(a.acoustics.noise_floor_db, "dB")}</td></tr>
            <tr><td>Wysokość głosu (F0)</td><td>{fmt(a.acoustics.mean_pitch_hz, "Hz")}</td></tr>
            <tr><td>Odchylenie F0</td><td>{fmt(a.acoustics.pitch_std_hz, "Hz")}</td></tr>
            <tr><td>Tempo mowy</td><td>{fmt(a.acoustics.speech_rate_wps, "sł/s")}</td></tr>
            <tr><td>Energia RMS</td><td>{fmt(a.acoustics.energy_rms, "")}</td></tr>
            <tr><td>Jitter</td><td>{fmt(a.acoustics.jitter, "")}</td></tr>
            <tr><td>Shimmer</td><td>{fmt(a.acoustics.shimmer, "")}</td></tr>
            <tr><td>Cisza</td><td>{fmtPct(a.acoustics.silence_ratio)}</td></tr>
            <tr><td>Nakładanie się</td><td>{fmtPct(a.acoustics.overlap_ratio)}</td></tr>
            <tr><td>Tło dźwiękowe</td><td>{a.acoustics.background_tags.join(", ") || "—"}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
