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

type Tab = "upload" | "progress" | "transcript" | "analysis" | "search" | "chat";

export default function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("upload");
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
      setActiveTab("progress");
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
        setActiveTab("upload");
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  const cancel = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.cancelSession(sessionId);
    } catch (e) {
      setError(String(e));
    }
  }, [sessionId]);

  const rerun = useCallback(
    async (phase: string) => {
      if (!sessionId) return;
      try {
        await api.rerunPhase(sessionId, phase);
      } catch (e) {
        setError(String(e));
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
      setActiveTab("transcript");
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
    setActiveTab("upload");
    ragIndex.clear();
  }, [sessionId, audioUrl, ragIndex]);

  const onSeek = useCallback((t: number) => {
    setSeekTo(t);
    setCurrentTime(t);
    setActiveTab("transcript");
  }, []);

  // Count errors in phases
  const errorCount = analysis
    ? Object.values(analysis.phases).filter((p) => p.status === "error").length
    : 0;
  const runningCount = analysis
    ? Object.values(analysis.phases).filter((p) => p.status === "running").length
    : 0;
  const hasTranscript = (analysis?.transcript.segments.length ?? 0) > 0;
  const hasResults = hasTranscript || (analysis?.speakers.length ?? 0) > 0;

  const tabs: { id: Tab; label: string; icon: string; disabled?: boolean; badge?: string }[] = [
    { id: "upload", label: "Nowa analiza", icon: "📤" },
    ...(analysis
      ? [
          {
            id: "progress" as Tab,
            label: "Postęp",
            icon: runningCount > 0 ? "⏳" : "📋",
            badge: errorCount > 0 ? String(errorCount) : runningCount > 0 ? "…" : undefined,
          },
          { id: "transcript" as Tab, label: "Transkrypt", icon: "📝", disabled: !hasTranscript },
          { id: "analysis" as Tab, label: "Analiza", icon: "📊", disabled: !hasResults },
          { id: "search" as Tab, label: "Szukaj", icon: "🔍", disabled: !ragIndex.size },
          { id: "chat" as Tab, label: "Chat", icon: "💬", disabled: !ragIndex.size },
        ]
      : []),
  ];

  return (
    <div className="app">
      <Toolbar sessionId={sessionId} analysis={analysis} onClear={clear} busy={busy} />

      {error && (
        <div className="error-bar">
          <span>⚠ {error}</span>
          <button className="ghost" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <nav className="tab-nav">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? " active" : ""}`}
            onClick={() => setActiveTab(t.id)}
            disabled={t.disabled}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
            {t.badge && (
              <span className={`tab-badge${t.badge === "…" ? " running" : ""}`}>{t.badge}</span>
            )}
          </button>
        ))}
      </nav>

      <div className={`tab-content${activeTab === "chat" ? " no-pad" : ""}`}>
        {activeTab === "upload" && (
          <UploadPanel onStart={start} onImport={importBundle} disabled={busy} />
        )}

        {activeTab === "progress" && analysis && (
          <ProgressPanel
            analysis={analysis}
            onRerun={rerun}
            onCancel={cancel}
            busy={busy}
          />
        )}

        {activeTab === "transcript" && analysis && (
          <div className="transcript-layout">
            <div>
              <TranscriptView
                analysis={analysis}
                audioUrl={audioUrl}
                currentTime={currentTime}
                onSeek={onSeek}
              />
            </div>
            <div>
              <Waveform
                audioUrl={audioUrl}
                onTime={setCurrentTime}
                seekTo={seekTo}
                currentTime={currentTime}
              />
            </div>
          </div>
        )}

        {activeTab === "analysis" && analysis && (
          <Dashboards analysis={analysis} onRerun={rerun} />
        )}

        {activeTab === "search" && analysis && (
          <SearchPanel index={ragIndex} onSeek={onSeek} analysis={analysis} />
        )}

        {activeTab === "chat" && analysis && (
          <ChatPanel index={ragIndex} analysis={analysis} />
        )}
      </div>
    </div>
  );
}
