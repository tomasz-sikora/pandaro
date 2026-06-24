"""
Tests for translator.translate_segments_to_polish.

Fixtures are derived from a real Russian prisoner-interview recording
(wywiad-ruski-jeniec.mp3) transcribed with Whisper large-v3.
Ollama HTTP calls are always mocked so the suite runs without a GPU/LLM.
"""
import copy
import json
from typing import List, Dict
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Real Russian-language chunks from the interview transcript
# (68 segments total, language="ru", 3 speakers)
# ---------------------------------------------------------------------------
RU_CHUNKS: List[Dict] = [
    {"id": 0,  "start": 17.11, "end": 36.30, "text": "добрый день как ваши дела почему так себе я потом задам вам вопрос вы будете с ним", "speaker": "GŁOS_01", "language": "ru"},
    {"id": 1,  "start": 36.30, "end": 47.82, "text": "разговаривать разговор будет записан и опубликован вы не говорите где вы", "speaker": "GŁOS_01", "language": "ru"},
    {"id": 2,  "start": 47.82, "end": 65.76, "text": "находитесь у меня все хорошо с нами относятся здесь хорошо кормят по и никто не трогать ждем", "speaker": "GŁOS_02", "language": "ru"},
    {"id": 3,  "start": 65.76, "end": 82.03, "text": "от замены своей это зависит от наших слушать кто занимается военнопленными там в россии как", "speaker": "GŁOS_02", "language": "ru"},
    {"id": 4,  "start": 83.13, "end": 86.49, "text": "Может быть мне куда-то подходить или кому-то позвонить, чтобы как-то не шевелились?", "speaker": "GŁOS_03", "language": "ru"},
    {"id": 5,  "start": 86.49, "end": 87.73, "text": "Да я даже не знаю, кому.", "speaker": "GŁOS_02", "language": "ru"},
    {"id": 6,  "start": 88.01, "end": 90.57, "text": "Ну если только в бригаде командиру сказать и всё.", "speaker": "GŁOS_02", "language": "ru"},
    {"id": 7,  "start": 91.38, "end": 92.24, "text": "А какой у вас там?", "speaker": "GŁOS_01", "language": "ru"},
    {"id": 8,  "start": 92.38, "end": 94.56, "text": "Ну в бригаде командир уже бегает от меня.", "speaker": "GŁOS_03", "language": "ru"},
    {"id": 9,  "start": 97.34, "end": 100.04, "text": "Ну, значит, так ему не нужен плен.", "speaker": "GŁOS_02", "language": "ru"},
    {"id": 10, "start": 105.86, "end": 107.82, "text": "Такой я проще, скорее бы домой.", "speaker": "GŁOS_03", "language": "ru"},
    {"id": 11, "start": 109.83, "end": 112.32, "text": "Ждём обмена.", "speaker": "GŁOS_02", "language": "ru"},
]

# Polish translations matching the 12 Russian chunks above (used as mock Ollama responses)
PL_TRANSLATIONS = [
    "Dzień dobry, jak się pan czuje? — Tak sobie. Potem zadam panu pytanie, będzie pan z nim",
    "rozmawiać. Rozmowa będzie nagrana i opublikowana. Proszę nie mówić, gdzie się pan",
    "znajduje. — U mnie wszystko dobrze, traktują nas tu dobrze, dobrze karmią i nikt nas nie rusza, czekamy",
    "na wymianę. To zależy od naszych. Słuchaj, kto zajmuje się jeńcami wojennymi tam w Rosji, jak",
    "Może powinienem do kogoś podejść albo do kogoś zadzwonić, żeby jakoś przestali się kręcić?",
    "Ja nawet nie wiem, do kogo.",
    "No chyba tylko dowódcy brygady powiedzieć i tyle.",
    "A jaki macie tam?",
    "No w brygadzie dowódca już ucieka przede mną.",
    "No to znaczy, że nie potrzebuje tego jeńca.",
    "Taki jestem prosty, szybciej do domu.",
    "Czekamy na wymianę.",
]


def _make_ollama_response(chunks: List[Dict]) -> str:
    """Build the JSON-structured response that mock Ollama returns for given chunks."""
    lines = [PL_TRANSLATIONS[c["id"]] for c in chunks if c["id"] < len(PL_TRANSLATIONS)]
    return json.dumps({"tlumaczenia_pl": lines})


def _fresh_chunks(source: List[Dict] = RU_CHUNKS) -> List[Dict]:
    """Deep-copy so mutations in one test don't bleed into others."""
    return copy.deepcopy(source)


# ---------------------------------------------------------------------------
# Helpers to import the module cleanly (avoids touching the global LRU cache)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_ollama_cache():
    """Clear the module-level LRU cache between tests."""
    from src import translator
    translator._ollama_cache._data.clear()
    yield
    translator._ollama_cache._data.clear()


# ===========================================================================
# 1. Polish passthrough — no Ollama call needed
# ===========================================================================
class TestPolishPassthrough:
    def test_all_chunks_get_text_pl_equal_to_text(self):
        from src.translator import translate_segments_to_polish

        chunks = [
            {"id": 0, "text": "Cześć, jak się masz?", "speaker": "GŁOS_01"},
            {"id": 1, "text": "Dobrze, dziękuję.", "speaker": "GŁOS_02"},
        ]
        result = translate_segments_to_polish(copy.deepcopy(chunks), "pl")

        for original, translated in zip(chunks, result):
            assert translated["text_pl"] == original["text"]

    def test_returns_same_list(self):
        from src.translator import translate_segments_to_polish

        chunks = [{"id": 0, "text": "Tekst po polsku.", "speaker": "GŁOS_01"}]
        result = translate_segments_to_polish(chunks, "pl")
        assert result is chunks  # mutates and returns the same list

    def test_no_ollama_call_for_polish(self):
        from src.translator import translate_segments_to_polish

        with patch("src.translator._call_ollama") as mock_ollama:
            translate_segments_to_polish(
                [{"id": 0, "text": "Polski tekst.", "speaker": "GŁOS_01"}], "pl"
            )
            mock_ollama.assert_not_called()


# ===========================================================================
# 2. Translation happy path — Russian interview chunks
# ===========================================================================
class TestTranslationHappyPath:
    def test_all_chunks_receive_text_pl(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks()
        mock_response = json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:len(chunks)]})

        with patch("src.translator._call_ollama", return_value=mock_response):
            result = translate_segments_to_polish(chunks, "ru")

        assert all("text_pl" in c for c in result)

    def test_translated_text_matches_mock_response(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks()
        mock_response = json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:len(chunks)]})

        with patch("src.translator._call_ollama", return_value=mock_response):
            result = translate_segments_to_polish(chunks, "ru")

        for i, chunk in enumerate(result):
            assert chunk["text_pl"] == PL_TRANSLATIONS[i], (
                f"chunk {i}: expected {PL_TRANSLATIONS[i]!r}, got {chunk['text_pl']!r}"
            )

    def test_original_text_field_is_preserved(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks()
        originals = [c["text"] for c in chunks]
        mock_response = json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:len(chunks)]})

        with patch("src.translator._call_ollama", return_value=mock_response):
            result = translate_segments_to_polish(chunks, "ru")

        for i, (chunk, orig) in enumerate(zip(result, originals)):
            assert chunk["text"] == orig, f"chunk {i} original text was mutated"

    def test_correct_language_name_in_prompt(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks(RU_CHUNKS[:2])
        captured_prompts = []

        def fake_ollama(prompt, **_):
            captured_prompts.append(prompt)
            return json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:2]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(chunks, "ru")

        assert len(captured_prompts) == 1
        assert "rosyjski" in captured_prompts[0]

    def test_unknown_language_code_used_verbatim_in_prompt(self):
        from src.translator import translate_segments_to_polish

        chunks = [{"id": 0, "text": "Some text.", "speaker": "GŁOS_01"}]
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": ["Jakiś tekst."]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(chunks, "xx")

        assert "xx" in captured[0]

    def test_invalid_json_response_falls_back_to_original(self):
        """When Ollama returns malformed JSON the original text is preserved."""
        from src.translator import translate_segments_to_polish

        chunks = [
            {"id": 0, "text": "Привет.", "speaker": "GŁOS_01"},
            {"id": 1, "text": "Пока.", "speaker": "GŁOS_02"},
        ]
        originals = [c["text"] for c in chunks]

        with patch("src.translator._call_ollama", return_value="not valid json at all"):
            result = translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        for chunk, orig in zip(result, originals):
            assert chunk["text_pl"] == orig

    def test_json_missing_translations_key_falls_back_to_original(self):
        from src.translator import translate_segments_to_polish

        chunks = [{"id": 0, "text": "Текст.", "speaker": "GŁOS_01"}]
        with patch("src.translator._call_ollama", return_value=json.dumps({"wrong_key": ["Tekst."]}) ):
            result = translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        assert result[0]["text_pl"] == "Текст."


# ===========================================================================
# 3. Fallback behaviour when Ollama fails
# ===========================================================================
class TestOllamaFailureFallback:
    def test_empty_ollama_response_falls_back_to_original(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks(RU_CHUNKS[:3])
        originals = [c["text"] for c in chunks]

        with patch("src.translator._call_ollama", return_value=""):
            result = translate_segments_to_polish(chunks, "ru")

        for chunk, orig in zip(result, originals):
            assert chunk["text_pl"] == orig

    def test_shorter_translations_array_falls_back_for_remainder(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks(RU_CHUNKS[:5])
        # Only 3 translations returned for 5 chunks
        partial_response = json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:3]})

        with patch("src.translator._call_ollama", return_value=partial_response):
            result = translate_segments_to_polish(chunks, "ru")

        # First 3 get translations
        for i in range(3):
            assert result[i]["text_pl"] == PL_TRANSLATIONS[i]
        # Last 2 fall back to original Russian text
        for i in range(3, 5):
            assert result[i]["text_pl"] == RU_CHUNKS[i]["text"]

    def test_all_chunks_get_text_pl_even_on_exception(self):
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks(RU_CHUNKS[:4])
        originals = [c["text"] for c in chunks]

        # _call_ollama itself swallows exceptions and returns ""; verify end-to-end
        with patch("src.translator._call_ollama", return_value=""):
            result = translate_segments_to_polish(chunks, "ru")

        assert len(result) == 4
        for chunk, orig in zip(result, originals):
            assert chunk["text_pl"] == orig


# ===========================================================================
# 4. Batching
# ===========================================================================
class TestBatching:
    def test_single_batch_for_short_transcript(self):
        """12 Russian chunks (~740 chars, under both 3000-char and 15-segment limits) → one call."""
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks()  # 12 segments
        call_count = 0

        def counting_ollama(prompt, **_):
            nonlocal call_count
            call_count += 1
            return json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:len(RU_CHUNKS)]})

        with patch("src.translator._call_ollama", side_effect=counting_ollama):
            translate_segments_to_polish(chunks, "ru")

        assert call_count == 1

    def test_multi_batch_splits_on_segment_count(self):
        """16 segments exceed BATCH_MAX_SEGMENTS=15, so two Ollama calls must be made."""
        from src.translator import translate_segments_to_polish

        short_text = "A " * 10   # tiny text so char limit is not the trigger
        big_chunks = [
            {"id": i, "text": short_text, "speaker": "GŁOS_01"}
            for i in range(16)
        ]
        call_count = 0

        def counting_ollama(prompt, **_):
            nonlocal call_count
            call_count += 1
            return json.dumps({"tlumaczenia_pl": ["Przetłumaczona linia."] * 16})

        with patch("src.translator._call_ollama", side_effect=counting_ollama):
            result = translate_segments_to_polish(copy.deepcopy(big_chunks), "en")

        assert call_count == 2
        assert all("text_pl" in c for c in result)

    def test_multi_batch_splits_on_char_count(self):
        """Two segments whose combined text exceeds BATCH_CHARS=3000 → two calls."""
        from src.translator import translate_segments_to_polish

        long_text = "Б " * 1600   # >3000 chars each segment
        big_chunks = [
            {"id": 0, "text": long_text, "speaker": "GŁOS_01"},
            {"id": 1, "text": long_text, "speaker": "GŁOS_02"},
        ]
        call_count = 0

        def counting_ollama(prompt, **_):
            nonlocal call_count
            call_count += 1
            return json.dumps({"tlumaczenia_pl": ["Tekst."]})

        with patch("src.translator._call_ollama", side_effect=counting_ollama):
            result = translate_segments_to_polish(copy.deepcopy(big_chunks), "ru")

        assert call_count == 2
        assert all("text_pl" in c for c in result)

    def test_empty_chunk_list_returns_empty(self):
        from src.translator import translate_segments_to_polish

        with patch("src.translator._call_ollama") as mock_ollama:
            result = translate_segments_to_polish([], "ru")

        assert result == []
        mock_ollama.assert_not_called()


# ===========================================================================
# 5. Prompt construction — source block integrity
# ===========================================================================
class TestPromptConstruction:
    def test_source_block_contains_speaker_tags(self):
        from src.translator import translate_segments_to_polish

        chunks = [
            {"id": 0, "text": "Привет.", "speaker": "GŁOS_01"},
            {"id": 1, "text": "Пока.", "speaker": "GŁOS_02"},
        ]
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": ["Cześć.", "Do widzenia."]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        # Lines are now numbered: "1. [GŁOS_01] Привет."
        assert "[GŁOS_01] Привет." in captured[0]
        assert "[GŁOS_02] Пока." in captured[0]

    def test_source_block_no_tag_for_empty_speaker(self):
        from src.translator import translate_segments_to_polish

        chunks = [{"id": 0, "text": "Просто текст.", "speaker": ""}]
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": ["Zwykły tekst."]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        assert "[] Просто текст." not in captured[0]
        assert "Просто текст." in captured[0]

    def test_prompt_requests_json_translations_array(self):
        from src.translator import translate_segments_to_polish

        chunks = [{"id": 0, "text": "Текст.", "speaker": "GŁOS_01"}]
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": ["Tekst."]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        assert "tlumaczenia_pl" in captured[0]
        assert "JSON" in captured[0]

    def test_prompt_contains_explicit_count(self):
        """Prompt must state the exact number of expected translations."""
        from src.translator import translate_segments_to_polish

        chunks = _fresh_chunks(RU_CHUNKS[:5])
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": PL_TRANSLATIONS[:5]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(chunks, "ru")

        assert "5" in captured[0]  # explicit count

    def test_source_lines_are_numbered(self):
        """Each input line must be prefixed with its 1-based sequence number."""
        from src.translator import translate_segments_to_polish

        chunks = [
            {"id": 0, "text": "Текст а.", "speaker": "GŁOS_01"},
            {"id": 1, "text": "Текст б.", "speaker": "GŁOS_02"},
            {"id": 2, "text": "Текст в.", "speaker": "GŁOS_01"},
        ]
        captured: list = []

        def fake_ollama(prompt, **_):
            captured.append(prompt)
            return json.dumps({"tlumaczenia_pl": ["Tekst a.", "Tekst b.", "Tekst c."]})

        with patch("src.translator._call_ollama", side_effect=fake_ollama):
            translate_segments_to_polish(copy.deepcopy(chunks), "ru")

        assert "1. " in captured[0]
        assert "2. " in captured[0]
        assert "3. " in captured[0]


# ===========================================================================
# 6. ollama_available helper
# ===========================================================================
class TestOllamaAvailable:
    def test_returns_true_on_200(self):
        from src.translator import ollama_available

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.translator.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            assert ollama_available() is True

    def test_returns_false_on_non_200(self):
        from src.translator import ollama_available

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("src.translator.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            assert ollama_available() is False

    def test_returns_false_on_connection_error(self):
        from src.translator import ollama_available

        with patch("src.translator.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.side_effect = Exception("refused")
            assert ollama_available() is False
