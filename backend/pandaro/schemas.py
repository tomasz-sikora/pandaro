"""Typed artifacts exchanged between the backend, the orchestrator and the SPA.

Every pipeline phase consumes/produces these pydantic models. They are the
contract with the frontend (which holds the canonical, ephemeral session state)
and with the export/import bundle.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Pipeline phase identifiers (the DAG)                                         #
# --------------------------------------------------------------------------- #
class Phase(str, Enum):
    INGEST = "ingest"
    VAD = "vad"
    ASR = "asr"
    ALIGN = "align"
    DIARIZE = "diarize"
    MERGE = "merge"
    SPEAKER_ID = "speaker_id"
    PARALINGUISTICS = "paralinguistics"
    ACOUSTICS = "acoustics"
    TRANSLATE = "translate"
    KEYWORDS = "keywords"
    SUMMARIZE = "summarize"
    RAG = "rag"
    REPORT = "report"


# Default execution order. Phases may be toggled off via the preset.
PHASE_ORDER: list[Phase] = [
    Phase.INGEST,
    Phase.VAD,
    Phase.ASR,
    Phase.ALIGN,
    Phase.DIARIZE,
    Phase.MERGE,
    Phase.SPEAKER_ID,
    Phase.PARALINGUISTICS,
    Phase.ACOUSTICS,
    Phase.TRANSLATE,
    Phase.KEYWORDS,
    Phase.SUMMARIZE,
    Phase.RAG,
    Phase.REPORT,
]


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Preset (user input before analysis)                                         #
# --------------------------------------------------------------------------- #
class SpeakerHint(BaseModel):
    name: str
    # Optional reference voiceprint (base64 wav) for speaker recognition.
    voiceprint_b64: str | None = None


class Preset(BaseModel):
    """User-provided settings captured before analysis (section 6 of the plan)."""

    languages: list[str] = Field(default_factory=lambda: ["pl"])
    expected_language: str = "pl"
    translate: bool = True
    translate_target: str = "pl"
    domain: str | None = None
    # Custom vocabulary / proper nouns to bias ASR + NER precision.
    vocabulary: list[str] = Field(default_factory=list)
    expected_speakers: int | None = None
    speaker_hints: list[SpeakerHint] = Field(default_factory=list)
    # Phase toggles.
    enabled_phases: list[Phase] = Field(default_factory=lambda: list(PHASE_ORDER))
    asr_backend: str | None = None
    quality: str = "best"  # best | balanced | fast
    confidence_threshold: float = 0.55
    summary_style: str = "bullet"  # bullet | narrative | minutes
    summary_target_language: str = "pl"


# --------------------------------------------------------------------------- #
# Transcript model                                                            #
# --------------------------------------------------------------------------- #
class Word(BaseModel):
    text: str
    start: float
    end: float
    # Aggregated 0..1 confidence (see text/confidence.py).
    confidence: float = 1.0
    low_confidence: bool = False


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    language: str | None = None
    words: list[Word] = Field(default_factory=list)
    # Raw Whisper signals, useful for the hallucination guard / UI.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    confidence: float = 1.0
    # Optional translation into the preset's target language.
    translation: str | None = None


class Transcript(BaseModel):
    language: str = "pl"
    duration: float = 0.0
    segments: list[Segment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)


# --------------------------------------------------------------------------- #
# Diarization / speakers                                                      #
# --------------------------------------------------------------------------- #
class SpeakerTurn(BaseModel):
    speaker: str
    start: float
    end: float


class SpeakerProfile(BaseModel):
    speaker: str
    name: str | None = None
    total_speech_s: float = 0.0
    # Paralinguistics (section 7).
    gender: str | None = None
    age: float | None = None
    dominant_emotion: str | None = None
    valence: float | None = None
    arousal: float | None = None
    dominance: float | None = None


# --------------------------------------------------------------------------- #
# Acoustic / OSINT fingerprint (section 8)                                    #
# --------------------------------------------------------------------------- #
class AcousticFeatures(BaseModel):
    snr_db: float | None = None
    noise_floor_db: float | None = None
    mean_pitch_hz: float | None = None
    pitch_std_hz: float | None = None
    speech_rate_wps: float | None = None  # words per second
    energy_rms: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    silence_ratio: float | None = None
    overlap_ratio: float | None = None
    background_tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# NLP outputs                                                                 #
# --------------------------------------------------------------------------- #
class Entity(BaseModel):
    text: str
    type: str  # PERSON | LOC | ORG | DATE | MISC
    count: int = 1


class Keyword(BaseModel):
    term: str
    score: float = 0.0


class Summary(BaseModel):
    overall: str = ""
    per_speaker: dict[str, str] = Field(default_factory=dict)
    topics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# RAG chunks (shipped to the browser WASM index)                              #
# --------------------------------------------------------------------------- #
class RagChunk(BaseModel):
    id: int
    text: str
    translation: str | None = None
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0
    # Slavic-aware search aids computed server-side (see text/*).
    normalized: str = ""
    phonetic: str = ""
    embedding: list[float] | None = None


# --------------------------------------------------------------------------- #
# Per-phase + whole-session state                                            #
# --------------------------------------------------------------------------- #
class PhaseState(BaseModel):
    phase: Phase
    status: PhaseStatus = PhaseStatus.PENDING
    progress: float = 0.0
    message: str | None = None
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None


class ConfidenceReport(BaseModel):
    mean_word_confidence: float = 1.0
    low_confidence_ratio: float = 0.0
    per_speaker: dict[str, float] = Field(default_factory=dict)


class Analysis(BaseModel):
    """The full, exportable analysis bundle for one recording."""

    version: str = "0.1.0"
    preset: Preset = Field(default_factory=Preset)
    media_duration: float = 0.0
    media_filename: str | None = None

    transcript: Transcript = Field(default_factory=Transcript)
    diarization: list[SpeakerTurn] = Field(default_factory=list)
    speakers: list[SpeakerProfile] = Field(default_factory=list)
    acoustics: AcousticFeatures = Field(default_factory=AcousticFeatures)
    entities: list[Entity] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)
    rag_chunks: list[RagChunk] = Field(default_factory=list)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)

    phases: dict[str, PhaseState] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
