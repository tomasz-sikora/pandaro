"""Tests for RRF fusion, cosine ranking, confidence and chunking logic."""

from pandaro.schemas import Segment, Transcript, Word
from pandaro.text import (
    annotate_transcript,
    chunk_transcript,
    cosine_rank,
    plan_summary_chunks,
    reciprocal_rank_fusion,
)
from pandaro.text.confidence import is_hallucination


def test_rrf_combines_lists():
    dense = [3, 1, 2]
    bm25 = [1, 2, 4]
    fused = reciprocal_rank_fusion([dense, bm25])
    ids = [doc for doc, _ in fused]
    # 1 appears high in both lists -> should win.
    assert ids[0] == 1
    assert set(ids) == {1, 2, 3, 4}


def test_rrf_weights_and_top_n():
    fused = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[2.0, 0.5], top_n=1)
    assert len(fused) == 1
    assert fused[0][0] == 1


def test_rrf_weight_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


def test_cosine_rank_orders_by_similarity():
    q = [1.0, 0.0]
    matrix = [[0.0, 1.0], [1.0, 0.1], [0.9, 0.0]]
    assert cosine_rank(q, matrix)[0] in (1, 2)


def test_confidence_annotation_flags_low_words():
    t = Transcript(
        segments=[
            Segment(
                id=0,
                start=0,
                end=1,
                text="cześć świat",
                speaker="SPEAKER_00",
                no_speech_prob=0.0,
                words=[
                    Word(text="cześć", start=0, end=0.5, confidence=0.9),
                    Word(text="świat", start=0.5, end=1.0, confidence=0.2),
                ],
            )
        ]
    )
    report = annotate_transcript(t, threshold=0.55)
    assert t.segments[0].words[0].low_confidence is False
    assert t.segments[0].words[1].low_confidence is True
    assert 0.0 < report.mean_word_confidence < 1.0
    assert report.low_confidence_ratio == 0.5
    assert "SPEAKER_00" in report.per_speaker


def test_hallucination_guard():
    seg = Segment(id=0, start=0, end=1, text="...", no_speech_prob=0.8, compression_ratio=3.0)
    assert is_hallucination(seg) is True
    ok = Segment(id=1, start=0, end=1, text="ok", no_speech_prob=0.1, compression_ratio=1.2)
    assert is_hallucination(ok) is False


def test_chunk_transcript_splits_on_speaker_change():
    t = Transcript(
        segments=[
            Segment(id=0, start=0, end=1, text="a", speaker="S0"),
            Segment(id=1, start=1, end=2, text="b", speaker="S0"),
            Segment(id=2, start=2, end=3, text="c", speaker="S1"),
        ]
    )
    chunks = chunk_transcript(t, max_tokens=1000, overlap_segments=0)
    assert len(chunks) == 2
    assert chunks[0].speaker == "S0"
    assert chunks[1].speaker == "S1"
    assert chunks[0].normalized != ""


def test_plan_summary_chunks_respects_budget():
    text = "\n".join(f"line {i}" for i in range(100))
    pieces = plan_summary_chunks(text, max_chars=50)
    assert len(pieces) > 1
    assert all(len(p) <= 60 for p in pieces)


def test_plan_summary_chunks_short_text():
    assert plan_summary_chunks("hello", max_chars=1000) == ["hello"]
    assert plan_summary_chunks("   ", max_chars=1000) == []
