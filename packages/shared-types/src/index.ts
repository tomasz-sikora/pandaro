// --- Audio / Transcript ---

export type SupportedLanguage = 'pl' | 'en' | 'ru' | 'uk' | 'de' | 'auto';

export interface SpeakerProfile {
  // audeering/wav2vec2-large-robust-24-ft-age-gender
  gender: string | null;       // 'zenski' | 'meski' | 'dziecko' | null
  gender_probs: Record<string, number> | null;
  age_estimate: number | null; // approximate age in years
  age_group: string | null;    // 'dziecko' | 'mlody' | 'dorosly' | 'starszy' | null
  confidence: number | null;   // 0-1
  // audeering/wav2vec2-large-robust-12-ft-emotion4
  emotion: string | null;
  emotion_probs: Record<string, number> | null;
  // speech rate & SNR (no model, always available)
  speech_rate_syllables_per_sec: number | null;
  speech_rate_label: string | null;
  snr_db: number | null;
  snr_label: string | null;
  /**
   * Human-readable display name identified by LLM from the transcript
   * (e.g. "Jan Kowalski") or a gender-based fallback (e.g. "Kobieta_1").
   * Undefined until the identification step completes.
   */
  display_name?: string | null;
}

export interface Word {
  text: string;       // includes leading space as returned by Whisper
  start: number;
  end: number;
  probability: number;
  alternatives?: string[];  // non-empty only for low-confidence words
}

export interface Segment {
  id: number;
  start: number;    // seconds
  end: number;      // seconds
  text: string;     // original language
  text_pl?: string; // Polish translation (same as text if already Polish)
  speaker: string;  // 'GŁOS_01', 'GŁOS_02', …
  language?: string;
  words?: Word[];           // word-level timestamps + per-word alternatives
  alternatives?: string[];  // legacy segment-level alternatives (compat)
}

// --- Entities ---

export interface Entities {
  persons: string[];        // bilingual: "Jan Kowalski (John Smith)"
  organizations: string[];
  locations: string[];
  dates: string[];
  keywords: string[];
}

// --- RAG ---

export interface VectorEntry {
  id: number;
  text: string;
  embedding: number[];
  metadata: {
    segmentIds: number[];
    start?: number;
    end?: number;
    speaker?: string;
  };
}

// --- Chat ---

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ text: string; score: number; start?: number; end?: number }>;
  createdAt: number;
}

// --- Session ---

export type ProcessingStep =
  | 'idle'
  | 'decoding'
  | 'loading_model'
  | 'transcribing'
  | 'diarizing'
  | 'profiling'
  | 'translating'
  | 'identifying'
  | 'extracting'
  | 'embedding'
  | 'summarizing'
  | 'done'
  | 'error';

export interface ProcessingState {
  step: ProcessingStep;
  progress: number; // 0-100
  message: string;
  error?: string;
}

export interface Session {
  id: string;
  fileName: string;
  fileSize: number;
  duration: number | null;
  detectedLanguage: string | null;
  processing: ProcessingState;
  segments: Segment[];
  speakerProfiles: Record<string, SpeakerProfile>;
  entities: Entities | null;
  summary: string | null;
  report: string | null;
  ragEntries: VectorEntry[];
  chat: ChatMessage[];
  createdAt: number;
  audioObjectUrl?: string | null;
  agentEvents: AgentEvent[];
  segmentsPartial?: boolean;
  qualityStats?: QualityStats | null;
  /** Segment quality scores from verify_transcript_quality (index → avg confidence) */
  segmentQuality?: Record<number, number>;
  /** Topics detected per time window */
  topics?: Array<{ start_sec: number; end_sec: number; topic: string }>;
  /** Active agent session ID (for hint injection) */
  agentSessionId?: string | null;
  /** Verbatim quotes and facts extracted by the agent */
  quotesAndFacts?: QuotesAndFacts | null;
}

// --- Quotes & Facts ---

export interface Quote {
  speaker: string;
  timestamp: string;
  text: string;
  significance?: string;
}

export interface Fact {
  speaker: string;
  text: string;
  category: 'number' | 'date' | 'name' | 'claim' | string;
}

export interface QuotesAndFacts {
  quotes: Quote[];
  facts: Fact[];
  decisions: Array<{ text: string; participants: string[] }>;
  key_questions: Array<{ speaker: string; text: string }>;
}

// --- Settings ---

export type AsrEngine = 'whisper' | 'vibevoice' | 'nemotron';

export interface Settings {
  transcribeUrl: string;
  ollamaUrl: string;
  ollamaModel: string;
  ollamaEmbeddingModel: string;
  useOllamaEmbeddings: boolean;
  whisperModel: string;
  sourceLanguage: SupportedLanguage;
  translateToPl: boolean;
  defaultAsrEngine: AsrEngine;
  theme: 'light' | 'dark' | 'system';
}

// --- Agent ---

export type AgentEventType =
  | 'agent_start'
  | 'agent_thinking'
  | 'tool_call'
  | 'tool_result'
  | 'tool_error'
  | 'agent_memory'
  | 'quality_report'
  | 'partial_segments'
  | 'segment_chunk'
  | 'translation_chunk'
  | 'translation_quality_check'
  | 'hint_injected'
  | 'progress'
  | 'result'
  | 'error';

export interface QualityStats {
  avg_confidence: number;
  low_confidence_count: number;
  repetitions: number;
  very_short_segments: number;
  long_gaps: number;
  total_segments: number;
  warnings?: string[];
  recommend_retranscribe?: boolean;
}

export interface AgentEvent {
  type: AgentEventType;
  /** For tool_call / tool_result / tool_error */
  tool?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  attempt?: number;
  success?: boolean;
  skipped?: boolean;
  /** For agent_thinking */
  step?: number;
  message?: string;
  /** For agent_memory */
  memory?: AgentMemory;
  /** For quality_report */
  avg_confidence?: number;
  low_confidence_segments?: Array<{ id: number; start?: number; text: string; confidence: number }>;
  warnings?: string[];
  stats?: QualityStats;
}

export interface AgentMemory {
  id: string;
  observation: string;
  improvement: string;
  tags: string[];
  created_at: number;
  times_applied: number;
}
