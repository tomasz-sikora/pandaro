from __future__ import annotations
from .context import AgentContext
from typing import Any, Dict, List
logger = __import__('logging').getLogger(__name__)

def _build_speaker_profiles(ctx: AgentContext) -> Dict:
    return {
        sp: {
            "gender": p.get("gender"),
            "gender_probs": p.get("gender_probs"),
            "age_estimate": p.get("age_estimate"),
            "age_group": p.get("age_group"),
            "confidence": p.get("confidence"),
            "display_name": ctx.display_names.get(sp),
            **ctx.audio_features_raw.get(sp, {}),
        }
        for sp, p in ctx.speaker_profiles_raw.items()
    }


def _build_segments_out(ctx: AgentContext) -> List[Dict]:
    result = []
    for i, s in enumerate(ctx.segments):
        text = s.get("text", "")
        # Never emit empty segments — they create phantom UI regions
        if not text or not text.strip():
            continue
        seg_out: Dict = {
            "id": i,
            "start": s.get("start"),
            "end": s.get("end"),
            "text": text,
            "text_pl": s.get("text_pl") or text,
            "speaker": s.get("speaker", f"GŁOS_{i+1:02d}"),
            "language": s.get("language") or ctx.detected_language,
            "words": s.get("words") or [],
        }
        # Preserve overlap annotation if present
        if s.get("overlapping"):
            seg_out["overlapping"] = True
            if s.get("overlap_with"):
                seg_out["overlap_with"] = s["overlap_with"]
            if s.get("overlap_sec") is not None:
                seg_out["overlap_sec"] = s["overlap_sec"]
        result.append(seg_out)
    # Re-index sequentially so IDs are contiguous in the UI
    for idx, seg in enumerate(result):
        seg["id"] = idx
    return result


# ── Main agent loop ───────────────────────────────────────────────────────────

