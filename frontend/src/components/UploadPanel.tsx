import { useRef, useState } from "react";
import { defaultPreset, PHASE_LABELS, type Preset } from "../types";

interface Props {
  onStart: (file: File, preset: Preset) => void;
  onImport: (file: File) => void;
  disabled: boolean;
}

const OPTIONAL_PHASES = [
  "diarize", "speaker_id", "paralinguistics", "acoustics",
  "translate", "keywords", "summarize", "rag",
];

export function UploadPanel({ onStart, onImport, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState<Preset>(defaultPreset());
  const [showPreset, setShowPreset] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const togglePhase = (phase: string) =>
    setPreset((p) => ({
      ...p,
      enabled_phases: p.enabled_phases.includes(phase)
        ? p.enabled_phases.filter((x) => x !== phase)
        : [...p.enabled_phases, phase],
    }));

  return (
    <div className="upload-layout">
      {/* Drop zone */}
      <div
        className={`drop-zone${dragging ? " over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <div className="drop-icon">🎙️</div>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>
          Przeciągnij nagranie lub kliknij
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          Obsługuje: MP3, WAV, M4A, OGG, MP4, WEBM i inne
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>

      {file && (
        <div className="file-chip">
          🎵 <strong>{file.name}</strong>{" "}
          <span className="muted">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
        </div>
      )}

      {/* Quick settings */}
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <h2 style={{ margin: 0, flex: 1 }}>Ustawienia analizy</h2>
          <button className="ghost" onClick={() => setShowPreset((s) => !s)}>
            {showPreset ? "▲ Zwiń" : "▼ Rozwiń"}
          </button>
        </div>

        <div className="form-grid">
          <div className="form-field">
            <label>Język nagrania</label>
            <select
              value={preset.expected_language}
              onChange={(e) => setPreset({ ...preset, expected_language: e.target.value })}
            >
              <option value="pl">Polski</option>
              <option value="uk">Ukraiński</option>
              <option value="ru">Rosyjski</option>
              <option value="en">Angielski</option>
              <option value="de">Niemiecki</option>
            </select>
          </div>
          <div className="form-field">
            <label>Styl podsumowania</label>
            <select
              value={preset.summary_style}
              onChange={(e) => setPreset({ ...preset, summary_style: e.target.value })}
            >
              <option value="bullet">Punkty</option>
              <option value="narrative">Narracja</option>
              <option value="minutes">Protokół</option>
            </select>
          </div>
        </div>

        {showPreset && (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="form-grid">
              <div className="form-field">
                <label>Liczba rozmówców (opcjonalnie)</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  placeholder="auto"
                  value={preset.expected_speakers ?? ""}
                  onChange={(e) =>
                    setPreset({
                      ...preset,
                      expected_speakers: e.target.value ? Number(e.target.value) : null,
                    })
                  }
                />
              </div>
              <div className="form-field">
                <label>Kontekst / dziedzina</label>
                <input
                  type="text"
                  placeholder="np. rozmowa handlowa, windykacja…"
                  value={preset.domain ?? ""}
                  onChange={(e) => setPreset({ ...preset, domain: e.target.value || null })}
                />
              </div>
            </div>

            <div className="form-field">
              <label>Słownik własny (nazwiska, terminy — przecinki)</label>
              <input
                type="text"
                placeholder="Kowalski, Warszawa, RODO…"
                onChange={(e) =>
                  setPreset({
                    ...preset,
                    vocabulary: e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
                  })
                }
              />
            </div>

            <div className="form-field">
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <input
                  type="checkbox"
                  id="translate-cb"
                  checked={preset.translate}
                  onChange={(e) => setPreset({ ...preset, translate: e.target.checked })}
                />
                <label htmlFor="translate-cb" style={{ cursor: "pointer", color: "var(--text)" }}>
                  Tłumacz na polski (UK/RU → PL)
                </label>
              </div>
            </div>

            <div className="form-field">
              <label>Aktywne etapy analizy</label>
              <div className="phase-toggles">
                {OPTIONAL_PHASES.map((ph) => {
                  const on = preset.enabled_phases.includes(ph);
                  return (
                    <label
                      key={ph}
                      className={`phase-toggle${on ? " on" : ""}`}
                    >
                      <input type="checkbox" checked={on} onChange={() => togglePhase(ph)} />
                      {PHASE_LABELS[ph] ?? ph}
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="upload-cta">
        <button
          className="primary"
          disabled={!file || disabled}
          onClick={() => file && onStart(file, preset)}
          style={{ flex: 1, justifyContent: "center" }}
        >
          {disabled ? <><span className="spinner" /> Przetwarzanie…</> : "▶ Uruchom analizę"}
        </button>
        <button onClick={() => importRef.current?.click()} disabled={disabled}>
          📂 Wczytaj .pandaro
        </button>
        <input
          ref={importRef}
          type="file"
          accept=".pandaro,application/json"
          hidden
          onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])}
        />
      </div>
    </div>
  );
}
