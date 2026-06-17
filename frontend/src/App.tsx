import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { Dashboards } from "./components/Dashboards";
import { ProgressPanel } from "./components/ProgressPanel";
import { SearchPanel } from "./components/SearchPanel";
import { Toolbar } from "./components/Toolbar";
import { TranscriptView } from "./components/TranscriptView";
import { UploadPanel } from "./components/UploadPanel";
import { Waveform } from "./components/Waveform";
import { registerEphemeralWipe, wipeBrowserStorage } from "./ephemeral";
import { RagIndex } from "./rag";
import type { Analysis, PhaseState, Preset } from "./types";

export default function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const ragIndex = useMemo(() => new RagIndex(), []);

  sessionIdRef.current = sessionId;

  // Efemeryczność: czyść sesję serwera i magazyn przeglądarki przy zamknięciu.
  useEffect(() => registerEphemeralWipe(() => sessionIdRef.current), []);

  // Odbuduj indeks RAG, gdy pojawią się fragmenty.
  useEffect(() => {
    if (analysis?.rag_chunks?.length) ragIndex.build(analysis.rag_chunks);
  }, [analysis?.rag_chunks, ragIndex]);

  const refresh = useCallback(async (sid: string) => {
    try {
      setAnalysis(await api.getSession(sid));
    } catch {
      /* sesja mogła zostać wyczyszczona */
    }
  }, []);

  const start = useCallback(
    async (file: File, preset: Preset) => {
      setBusy(true);
      setError(null);
      try {
        setAudioUrl(URL.createObjectURL(file));
        const { session_id, analysis: a } = await api.createSession(
          file,
          JSON.stringify(preset)
        );
        setSessionId(session_id);
        setAnalysis(a);

        wsRef.current?.close();
        wsRef.current = api.openProgressSocket(session_id, (st: PhaseState) => {
          setAnalysis((prev) =>
            prev ? { ...prev, phases: { ...prev.phases, [st.phase]: st } } : prev
          );
          if (st.status === "done" || st.status === "error") refresh(session_id);
        });
        await api.run(session_id);
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  const rerun = useCallback(
    async (phase: string) => {
      if (!sessionId) return;
      setBusy(true);
      try {
        await api.rerunPhase(sessionId, phase);
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [sessionId]
  );

  const importBundle = useCallback(async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const { session_id, analysis: a } = await api.importBundle(file);
      setSessionId(session_id);
      setAnalysis(a);
      setAudioUrl(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const clear = useCallback(async () => {
    if (sessionId) await api.clearSession(sessionId);
    await wipeBrowserStorage();
    wsRef.current?.close();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAnalysis(null);
    setSessionId(null);
    setAudioUrl(null);
    setCurrentTime(0);
    setSeekTo(null);
  }, [sessionId, audioUrl]);

  const onSeek = useCallback((t: number) => {
    setSeekTo(t);
    setCurrentTime(t);
  }, []);

  return (
    <div className="app">
      <Toolbar sessionId={sessionId} analysis={analysis} onClear={clear} busy={busy} />
      {error && <div className="error-bar">{error}</div>}

      <main className="layout">
        <section className="col col-left">
          <UploadPanel onStart={start} onImport={importBundle} disabled={busy} />
          {analysis && <ProgressPanel analysis={analysis} onRerun={rerun} busy={busy} />}
          {analysis && <Dashboards analysis={analysis} />}
        </section>

        <section className="col col-center">
          <Waveform audioUrl={audioUrl} onTime={setCurrentTime} seekTo={seekTo} />
          {analysis && (
            <TranscriptView
              analysis={analysis}
              audioUrl={audioUrl}
              currentTime={currentTime}
              onSeek={onSeek}
            />
          )}
        </section>

        <section className="col col-right">
          {analysis && <SearchPanel index={ragIndex} onSeek={onSeek} />}
          {analysis && <ChatPanel index={ragIndex} analysis={analysis} />}
        </section>
      </main>
    </div>
  );
}
