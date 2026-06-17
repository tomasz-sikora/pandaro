import { useRef, useState } from "react";
import { defaultPreset, PHASE_LABELS, type Preset } from "../types";

interface Props {
  onStart: (file: File, preset: Preset) => void;
  onImport: (file: File) => void;
  disabled: boolean;
}

const OPTIONAL_PHASES = [
  "diarize",
  "speaker_id",
  "paralinguistics",
  "acoustics",
  "translate",
  "keywords",
  "summarize",
  "rag",
];

export function UploadPanel({ onStart, onImport, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState<Preset>(defaultPreset());
  const [open, setOpen] = useState(true);
  const importRef = useRef<HTMLInputElement>(null);

  const togglePhase = (phase: string) => {
    setPreset((p) => ({
      ...p,
      enabled_phases: p.enabled_phases.includes(phase)
        ? p.enabled_phases.filter((x) => x !== phase)
        : [...p.enabled_phases, phase],
    }));
  };

  return (
    <div className="card">
      <h2>Nowa analiza</h2>
      <div className="row">
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          disabled={disabled}
        />
        <button
          className="primary"
          disabled={!file || disabled}
          onClick={() => file && onStart(file, preset)}
        >
          Analizuj
        </button>
        <button onClick={() => importRef.current?.click()} disabled={disabled}>
          Wczytaj raport (.pandaro)
        </button>
        <input
          ref={importRef}
          type="file"
          accept=".pandaro,application/json"
          hidden
          onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])}
        />
      </div>

      <button className="link" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} Ustawienia analizy
      </button>

      {open && (
        <div className="preset">
          <label>
            Język nagrania
            <select
              value={preset.expected_language}
              onChange={(e) =>
                setPreset({ ...preset, expected_language: e.target.value })
              }
            >
              <option value="pl">polski</option>
              <option value="uk">ukraiński</option>
              <option value="ru">rosyjski</option>
              <option value="en">angielski</option>
            </select>
          </label>

          <label className="check">
            <input
              type="checkbox"
              checked={preset.translate}
              onChange={(e) => setPreset({ ...preset, translate: e.target.checked })}
            />
            Tłumacz na polski (UK/RU → PL)
          </label>

          <label>
            Liczba rozmówców (opcjonalnie)
            <input
              type="number"
              min={1}
              max={20}
              value={preset.expected_speakers ?? ""}
              onChange={(e) =>
                setPreset({
                  ...preset,
                  expected_speakers: e.target.value ? Number(e.target.value) : null,
                })
              }
            />
          </label>

          <label>
            Kontekst / dziedzina
            <input
              type="text"
              placeholder="np. rozmowa handlowa, windykacja…"
              value={preset.domain ?? ""}
              onChange={(e) => setPreset({ ...preset, domain: e.target.value || null })}
            />
          </label>

          <label>
            Słownik (nazwiska, terminy — przecinki)
            <input
              type="text"
              placeholder="Kowalski, Warszawa, RODO…"
              onChange={(e) =>
                setPreset({
                  ...preset,
                  vocabulary: e.target.value
                    .split(",")
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          </label>

          <label>
            Styl podsumowania
            <select
              value={preset.summary_style}
              onChange={(e) => setPreset({ ...preset, summary_style: e.target.value })}
            >
              <option value="bullet">punkty</option>
              <option value="narrative">narracja</option>
              <option value="minutes">protokół</option>
            </select>
          </label>

          <fieldset>
            <legend>Etapy analizy</legend>
            {OPTIONAL_PHASES.map((ph) => (
              <label key={ph} className="check">
                <input
                  type="checkbox"
                  checked={preset.enabled_phases.includes(ph)}
                  onChange={() => togglePhase(ph)}
                />
                {PHASE_LABELS[ph]}
              </label>
            ))}
          </fieldset>
        </div>
      )}
    </div>
  );
}
