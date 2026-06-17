import { useEffect, useRef } from "react";
import { formatTime } from "../text";
import type { Analysis } from "../types";

interface Props {
  analysis: Analysis;
  audioUrl: string | null;
  currentTime: number;
  onSeek: (t: number) => void;
}

// Kolory rozmówców (stabilne wg etykiety).
const PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
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

  return (
    <div className="card transcript">
      <h2>Transkrypt</h2>
      <p className="hint">
        Słowa o niskiej pewności są podkreślone. Kliknij segment, aby przejść do nagrania.
      </p>
      <div className="segments">
        {segments.map((seg) => {
          const active = currentTime >= seg.start && currentTime < seg.end;
          return (
            <div
              key={seg.id}
              ref={active ? activeRef : undefined}
              className={`segment ${active ? "active" : ""}`}
              onClick={() => onSeek(seg.start)}
            >
              <span className="seg-time">{formatTime(seg.start)}</span>
              {seg.speaker && (
                <span className="seg-speaker" style={{ color: speakerColor(seg.speaker) }}>
                  {seg.speaker}
                </span>
              )}
              <span className="seg-text">
                {seg.words.length
                  ? seg.words.map((w, i) => (
                      <span key={i} className={w.low_confidence ? "low-conf" : ""}>
                        {w.text}{" "}
                      </span>
                    ))
                  : seg.text}
              </span>
              {seg.translation && <span className="seg-translation">↳ {seg.translation}</span>}
            </div>
          );
        })}
        {!segments.length && <p className="muted">Brak transkryptu.</p>}
      </div>
    </div>
  );
}
