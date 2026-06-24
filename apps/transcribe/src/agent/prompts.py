from __future__ import annotations
from .context import AgentContext
from typing import Dict, List
from ..memory import format_memories_for_prompt, load_memories
import logging
logger = logging.getLogger(__name__)

# ── Tool schemas (compact; one entry per line for readability) ────────────────
# Keep this list in sync with _TOOL_IMPL in tools.py.

TOOL_SCHEMAS: List[Dict] = [
    # Calibration
    {"type": "function", "function": {"name": "get_audio_info", "description": "Decode audio to PCM and return duration, language hint, translate flag. Call FIRST.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "analyze_audio_quality", "description": "Report RMS, dynamic range, clipping, silence ratio, noise floor. dynamic_range_db < 12 means phone-call audio.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "detect_noise_regions", "description": "Find silence/noise time ranges (used for the UI waveform overlay).", "parameters": {"type": "object", "properties": {"min_silence_sec": {"type": "number"}, "energy_threshold": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "probe_audio_fragment", "description": "Transcribe 2-3 short fragments to detect language and estimate quality. Use only when language is 'auto'.", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "duration_sec": {"type": "number"}, "language": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "detect_speaker_count", "description": "Estimate number of speakers via pyannote (full audio for recordings <=15 min).", "parameters": {"type": "object", "properties": {"duration_sec": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "set_transcription_params", "description": "Store Whisper params for the next transcribe call. Fields: vad_filter_threshold, beam_size, temperature, no_speech_threshold, compression_ratio_threshold, chunk_minutes, overlap_seconds, max_ctx_segments.", "parameters": {"type": "object", "properties": {"vad_filter_threshold": {"type": "number"}, "beam_size": {"type": "integer"}, "temperature": {"type": "number"}, "no_speech_threshold": {"type": "number"}, "compression_ratio_threshold": {"type": "number"}, "chunk_minutes": {"type": "number"}, "overlap_seconds": {"type": "number"}, "max_ctx_segments": {"type": "integer"}}, "required": []}}},
    # Transcription
    {"type": "function", "function": {"name": "diarize_first_transcribe", "description": "PREFERRED. Pyannote diarizes first, then Whisper transcribes each speaker turn. Captures short interjections, filters hallucinations by confidence, annotates overlaps. Does diarization internally.", "parameters": {"type": "object", "properties": {"num_speakers": {"type": "integer"}, "language": {"type": "string"}, "min_confidence": {"type": "number", "description": "Drop turns below this avg word confidence (default 0.30)."}, "start_sec": {"type": "number", "description": "Optional: re-process only this time range."}, "end_sec": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "transcribe_audio", "description": "Plain full-audio ASR (no per-turn diarization). Use only for single-speaker audio or the vibevoice engine.", "parameters": {"type": "object", "properties": {"engine": {"type": "string", "enum": ["whisper", "vibevoice"]}, "language": {"type": "string"}}, "required": ["engine"]}}},
    {"type": "function", "function": {"name": "verify_transcript_quality", "description": "Report avg confidence, repetitions, gaps, low-confidence segments. Returns recommend_retranscribe.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "tag_segments", "description": "Tag segments: interjection, question, low-conf, silence-gap, overlap.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "merge_short_segments", "description": "Merge same-speaker segments separated by tiny gaps. Conservative (gap_sec default 0.05).", "parameters": {"type": "object", "properties": {"gap_sec": {"type": "number"}, "min_duration_sec": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "retranscribe_time_range", "description": "Re-transcribe start_sec..end_sec and replace those segments. Used for UI fragment re-processing.", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}, "language": {"type": "string"}, "params": {"type": "object"}}, "required": ["start_sec", "end_sec"]}}},
    # Diarization (only needed if you used plain transcribe_audio)
    {"type": "function", "function": {"name": "diarize_audio", "description": "Assign speaker labels to existing segments via pyannote. Only needed after transcribe_audio.", "parameters": {"type": "object", "properties": {"num_speakers": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "refine_speaker_assignments", "description": "Fix short/interjection speaker errors via neighbour voting.", "parameters": {"type": "object", "properties": {"gap_sec": {"type": "number"}, "short_sec": {"type": "number"}, "window": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "merge_duplicate_speakers", "description": "Merge speaker IDs that are the SAME physical person (pyannote sometimes over-splits one speaker). MFCC similarity, very conservative. Merging is destructive and loses real speakers, so ONLY call this if you suspect over-splitting. NEVER pass similarity_threshold below 0.99 — phone/codec compression makes different voices look ~0.98 similar.", "parameters": {"type": "object", "properties": {"similarity_threshold": {"type": "number"}, "min_duration_sec": {"type": "number"}}, "required": []}}},
    {"type": "function", "function": {"name": "profile_speakers", "description": "Extract gender, age, emotion, speech rate, SNR per speaker.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "identify_speakers", "description": "Infer real names/roles from context and push display names to the UI.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    # Translation
    {"type": "function", "function": {"name": "translate_to_polish", "description": "Translate non-Polish segments to Polish. ONLY call if translate is requested AND language != pl.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "validate_translation_quality", "description": "LLM quality score 1-5 on a sample of translations.", "parameters": {"type": "object", "properties": {"sample_size": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "retranslate_segments", "description": "Re-translate specific segment IDs with improved context.", "parameters": {"type": "object", "properties": {"segment_ids": {"type": "array", "items": {"type": "integer"}}, "temperature": {"type": "number"}}, "required": ["segment_ids"]}}},
    # Analysis & synthesis
    {"type": "function", "function": {"name": "emit_partial_result", "description": "Push the current transcript and speaker profiles to the UI immediately.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "extract_entities", "description": "Extract persons, organisations, locations, dates, keywords.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "extract_quotes_and_facts", "description": "Extract key verbatim quotes, facts, decisions. Skip for recordings < 5 min.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "detect_topics", "description": "Topic label per time window (chapter markers). Skip for recordings < 10 min.", "parameters": {"type": "object", "properties": {"window_minutes": {"type": "number"}, "max_windows": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "build_rag_index", "description": "Generate embeddings for chat/search. Non-fatal if embeddings unavailable.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "summarize_transcript", "description": "Structured Polish summary. style: brief | detailed | structured.", "parameters": {"type": "object", "properties": {"style": {"type": "string", "enum": ["brief", "detailed", "structured"]}}, "required": []}}},
    # Memory & control
    {"type": "function", "function": {"name": "save_memory", "description": "Persist a useful observation + improvement for future sessions.", "parameters": {"type": "object", "properties": {"observation": {"type": "string"}, "improvement": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["observation", "improvement"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Signal processing is complete. Call ONLY after the transcript and all requested analysis are done.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": []}}},
]


def _build_system_prompt(ctx: AgentContext) -> str:
    """
    System prompt for gemma4:26b tool-calling.

    Kept STATIC (no session id / timestamp) so Ollama can reuse the evaluated
    KV prefix across sessions. Per-session details go in the first user message.
    """
    memories_block = format_memories_for_prompt(load_memories())

    return (
        "You are Pandaro, an expert audio transcription agent.\n\n"
        "HOW TO ACT:\n"
        "- Call EXACTLY ONE tool per response. Do not write prose.\n"
        "- Use tool names exactly as given. When done, call finish.\n"
        "- Read each tool result before deciding the next step.\n"
        "- Respond in Polish in any message/emit fields.\n\n"
        "DEFAULT WORKFLOW (adapt based on results):\n"
        "1. get_audio_info — always first.\n"
        "2. analyze_audio_quality — note dynamic_range_db (<12 = phone call → beam_size=7).\n"
        "3. detect_noise_regions — feeds the UI waveform.\n"
        "4. probe_audio_fragment — ONLY if language is 'auto'.\n"
        "5. detect_speaker_count.\n"
        "6. diarize_first_transcribe(num_speakers=N, language=...) — THE main step.\n"
        "   It diarizes + transcribes + filters hallucinations + annotates overlaps.\n"
        "   After it, do NOT call diarize_audio or transcribe_audio.\n"
        "7. tag_segments.\n"
        "8. merge_duplicate_speakers — ONLY if pyannote clearly over-split one speaker; keep default threshold (never below 0.99). Prefer NOT calling it.\n"
        "9. profile_speakers, then identify_speakers.\n"
        "10. translate_to_polish — ONLY if translation was requested AND language != pl.\n"
        "    If translation is not requested, SKIP it.\n"
        "11. emit_partial_result — show the user a live preview.\n"
        "12. extract_entities.\n"
        "13. extract_quotes_and_facts — skip if recording < 5 min.\n"
        "14. detect_topics — skip if recording < 10 min.\n"
        "15. build_rag_index.\n"
        "16. summarize_transcript(style='structured').\n"
        "17. save_memory (optional), then finish.\n\n"
        "QUALITY RULES:\n"
        "- If verify_transcript_quality recommends it, retranscribe_time_range on the worst span.\n"
        "- Short single-word turns (tak/nie/mhm/dobra/halo) are valid interjections — keep them.\n"
        "- For single-speaker audio you may use transcribe_audio + diarize_audio instead of step 6.\n"
    ) + memories_block
