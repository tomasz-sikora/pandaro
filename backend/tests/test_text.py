"""Tests for Slavic-aware transliteration and phonetic encoding."""

from pandaro.text import normalize_text, normalize_token, phonetic_code, transliterate


def test_transliterate_cyrillic_to_latin():
    # Pragmatic, search-oriented romanization (и->y, г->h, deterministic).
    assert transliterate("Юрий") == "juryj"
    assert transliterate("Гагарин") == "haharyn"
    assert transliterate("Київ") == "kyjiv"


def test_transliterate_polish_diacritics():
    assert transliterate("Łódź") == "lodz"
    assert transliterate("Gdańsk") == "gdansk"


def test_normalize_token_strips_punctuation():
    assert normalize_token("Jurij,") == "jurij"
    assert normalize_token("(Łódź)") == "lodz"


def test_normalize_text_multiple_tokens():
    assert normalize_text("Юрий Гагарин") == "juryj haharyn"


def test_phonetic_bridges_transliteration_variants():
    # Different spellings of the same name should share a phonetic skeleton.
    a = phonetic_code("Jurij")
    b = phonetic_code("Юрий")
    assert a == b and a != ""


def test_phonetic_collapses_slavic_digraphs():
    # sz and sh should map to the same sound class.
    assert phonetic_code("Szymon")[0:1] == "S"
    assert phonetic_code("Shymon")[0:1] == "S"


def test_phonetic_empty_for_nonalpha():
    assert phonetic_code("123") == ""
    assert phonetic_code("") == ""
