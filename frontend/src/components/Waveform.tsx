import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

interface Props {
  audioUrl: string | null;
  onTime: (t: number) => void;
  seekTo: number | null;
}

// Fala dźwiękowa + odtwarzacz. Klik w transkrypcie ustawia `seekTo`.
export function Waveform({ audioUrl, onTime, seekTo }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#cbd5e1",
      progressColor: "#3b82f6",
      cursorColor: "#1e293b",
      height: 80,
      normalize: true,
    });
    ws.load(audioUrl);
    ws.on("audioprocess", (t: number) => onTime(t));
    ws.on("seeking", (t: number) => onTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [audioUrl, onTime]);

  useEffect(() => {
    const ws = wsRef.current;
    if (ws && seekTo != null && ws.getDuration() > 0) {
      ws.setTime(seekTo);
    }
  }, [seekTo]);

  if (!audioUrl) return null;

  return (
    <div className="card waveform">
      <div className="row">
        <button onClick={() => wsRef.current?.playPause()}>
          {playing ? "⏸ Pauza" : "▶ Odtwórz"}
        </button>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
