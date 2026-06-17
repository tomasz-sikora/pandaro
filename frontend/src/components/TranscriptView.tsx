import { useEffect, useRef } from "react";
import { formatTime } from "../text";
import type { Analysis } from "../types";

interface Props {
  analysis: Analysis;
  audioUrl: string | null;
  currentTime: number;
  onSeek: (t: number) => void;
}

const PALETTE = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#ec4899","#14b8a6","#f97316"];
export function speakerColor(speaker: string | null): string {
  if (!speaker) return "#94a3b8";
  const n = parseInt(speaker.replace(/\D/g, "") || "0", 10);
  return PALETTE[n % PALETTE.length];
}

export function TranscriptView({ analysis, currentTime, onSeek }: Props) {
  const activeRef = useRef<HTMLDivElement>(null);
  const segments = analysis.transcript.segments;

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [currentTime]);

  if (!segments.length) {
    return (
      <div className="card">
        <h2>Transkrypt</h2>
        <div className="empty">Brak transkryptu — uruchom fazę ASR.</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>Transkrypt</h2>
        <span className="hint">
          {segments.length} segmentów · {formatTime(analysis.transcript.duration)} ·{" "}
          język: {analysis.transcript.language}
        </span>
      </div>
      <p className="hint" style={{ marginBottom: 8 }}>
        Podkreślone słowa mają niską pewność ASR. Kliknij segment → przeskok do nagrania.
      </p>
      <div className="segments">
        {segments.map((seg) => {
          const active = currentTime >= seg.start && currentTime < seg.end;
          const color = speakerColor(seg.speaker);
          return (
            <div
              key={seg.id}
              ref={active ? activeRef : undefined}
              className={`segment ${active ? "active" : ""}`}
              onClick={() => onSeek(seg.start)}
            >
              <span className="seg-time">{formatTime(seg.start)}</span>
              {seg.speaker && (
                <span className="seg-dot" style={{ background: color }} title={seg.speaker} />
              )}
              <div className="seg-body">
                {seg.speaker && (
                  <div className="seg-speaker" style={{ color }}>
                    {seg.speaker}
                  </div>
                )}
                <span className="seg-text">
                  {seg.words.length
                    ? seg.words.map((w, i) => (
                        <span
                          key={i}
                          className={`word${w.low_confidence ? " lc" : ""}`}
                          title={w.low_confidence ? `Pewność: ${Math.round(w.confidence * 100)}%` : undefined}
                        >
                          {w.text}{" "}
                        </span>
                      ))
                    : seg.text}
                </span>
                {seg.translation && (
                  <span className="seg-translation">↳ {seg.translation}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
