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
  /** Blob object URL created from the uploaded audio file. */
  audioObjectUrl?: string | null;
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
