// Typy współdzielone z backendem (podzbiór pandaro.schemas).

export interface Word {
  text: string;
  start: number;
  end: number;
  confidence: number;
  low_confidence: boolean;
}

export interface Segment {
  id: number;
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  language: string | null;
  words: Word[];
  confidence: number;
  translation: string | null;
  no_speech_prob: number | null;
}

export interface Transcript {
  language: string;
  duration: number;
  segments: Segment[];
}

export interface SpeakerTurn {
  speaker: string;
  start: number;
  end: number;
}

export interface SpeakerProfile {
  speaker: string;
  name: string | null;
  total_speech_s: number;
  gender: string | null;
  age: number | null;
  dominant_emotion: string | null;
  valence: number | null;
  arousal: number | null;
  dominance: number | null;
}

export interface AcousticFeatures {
  snr_db: number | null;
  noise_floor_db: number | null;
  mean_pitch_hz: number | null;
  pitch_std_hz: number | null;
  speech_rate_wps: number | null;
  energy_rms: number | null;
  jitter: number | null;
  shimmer: number | null;
  silence_ratio: number | null;
  overlap_ratio: number | null;
  background_tags: string[];
}

export interface Entity {
  text: string;
  type: string;
  count: number;
}

export interface Keyword {
  term: string;
  score: number;
}

export interface Summary {
  overall: string;
  per_speaker: Record<string, string>;
  topics: string[];
}

export interface RagChunk {
  id: number;
  text: string;
  translation: string | null;
  speaker: string | null;
  start: number;
  end: number;
  confidence: number;
  normalized: string;
  phonetic: string;
  embedding: number[] | null;
}

export interface ConfidenceReport {
  mean_word_confidence: number;
  low_confidence_ratio: number;
  per_speaker: Record<string, number>;
}

export type PhaseStatus = "pending" | "running" | "done" | "skipped" | "error";

export interface PhaseState {
  phase: string;
  status: PhaseStatus;
  progress: number;
  message: string | null;
  error: string | null;
}

export interface Preset {
  languages: string[];
  expected_language: string;
  translate: boolean;
  translate_target: string;
  domain: string | null;
  vocabulary: string[];
  expected_speakers: number | null;
  enabled_phases: string[];
  asr_backend: string | null;
  quality: string;
  confidence_threshold: number;
  summary_style: string;
  summary_target_language: string;
}

export interface Analysis {
  version: string;
  preset: Preset;
  media_duration: number;
  media_filename: string | null;
  transcript: Transcript;
  diarization: SpeakerTurn[];
  speakers: SpeakerProfile[];
  acoustics: AcousticFeatures;
  entities: Entity[];
  keywords: Keyword[];
  summary: Summary;
  rag_chunks: RagChunk[];
  confidence: ConfidenceReport;
  phases: Record<string, PhaseState>;
  model_versions: Record<string, string>;
}

export const PHASE_LABELS: Record<string, string> = {
  ingest: "Wczytywanie",
  vad: "Detekcja mowy (VAD)",
  asr: "Transkrypcja",
  align: "Wyrównanie słów",
  diarize: "Diaryzacja (rozmówcy)",
  merge: "Łączenie z rozmówcami",
  speaker_id: "Rozpoznawanie rozmówców",
  paralinguistics: "Analiza głosu (wiek/płeć/emocje)",
  acoustics: "Cechy akustyczne / OSINT",
  translate: "Tłumaczenie",
  keywords: "Słowa kluczowe i encje",
  summarize: "Podsumowanie",
  rag: "Budowa indeksu RAG",
  report: "Raport",
};

export function defaultPreset(): Preset {
  return {
    languages: ["pl"],
    expected_language: "pl",
    translate: true,
    translate_target: "pl",
    domain: null,
    vocabulary: [],
    expected_speakers: null,
    enabled_phases: [
      "ingest",
      "vad",
      "asr",
      "align",
      "diarize",
      "merge",
      "speaker_id",
      "paralinguistics",
      "acoustics",
      "translate",
      "keywords",
      "summarize",
      "rag",
      "report",
    ],
    asr_backend: null,
    quality: "best",
    confidence_threshold: 0.55,
    summary_style: "bullet",
    summary_target_language: "pl",
  };
}
