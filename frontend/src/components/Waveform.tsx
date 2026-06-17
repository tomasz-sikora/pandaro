import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { formatTime } from "../text";

interface Props {
  audioUrl: string | null;
  onTime: (t: number) => void;
  seekTo: number | null;
  currentTime?: number;
}

export function Waveform({ audioUrl, onTime, seekTo, currentTime = 0 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#334155",
      progressColor: "#38bdf8",
      cursorColor: "#60a5fa",
      height: 64,
      normalize: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
    });
    ws.load(audioUrl);
    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("audioprocess", (t: number) => onTime(t));
    ws.on("seeking", (t: number) => onTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; };
  }, [audioUrl, onTime]);

  useEffect(() => {
    const ws = wsRef.current;
    if (ws && seekTo != null && ws.getDuration() > 0) ws.setTime(seekTo);
  }, [seekTo]);

  if (!audioUrl) return null;

  return (
    <div className="card waveform-card">
      <div className="waveform-controls">
        <button className="icon" onClick={() => wsRef.current?.playPause()} title={playing ? "Pauza" : "Odtwórz"}>
          {playing ? "⏸" : "▶"}
        </button>
        <button className="icon" onClick={() => wsRef.current?.stop()} title="Stop">⏹</button>
        <button className="icon ghost" onClick={() => wsRef.current?.skip(-10)} title="−10s">⏪</button>
        <button className="icon ghost" onClick={() => wsRef.current?.skip(10)} title="+10s">⏩</button>
        <span className="waveform-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
