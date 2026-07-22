"""
Tests for core.text_normalizer
"""
import pytest
from core.text_normalizer import TextNormalizer


@pytest.fixture
def normalizer():
    return TextNormalizer.get()


class TestArabicNormalization:
    def test_diacritics_removed(self, normalizer):
        result = normalizer.process("إِنَّ ٱلْعِلْمَ نُورٌ", language="ar")
        assert "ِ" not in result.normalized_text
        assert "َ" not in result.normalized_text
        assert "ٌ" not in result.normalized_text

    def test_alef_normalized(self, normalizer):
        result = normalizer.process("إسلام أحمد آمنة", language="ar")
        assert "إ" not in result.normalized_text
        assert "أ" not in result.normalized_text
        assert "آ" not in result.normalized_text
        assert "ا" in result.normalized_text

    def test_tatweel_removed(self, normalizer):
        result = normalizer.process("البحـــث الدلالـــي", language="ar")
        assert "ـ" not in result.normalized_text

    def test_arabic_digits_to_western(self, normalizer):
        result = normalizer.process("الفصل ٣ من ٢٠٢٤", language="ar")
        assert "3" in result.normalized_text
        assert "2024" in result.normalized_text

    def test_search_text_has_variants(self, normalizer):
        result = normalizer.process("المحاضرة الأولى", language="ar")
        assert result.search_text
        assert len(result.search_text) > 0


class TestLatinNormalization:
    def test_lowercased(self, normalizer):
        result = normalizer.process("Hello WORLD", language="en")
        assert result.normalized_text == "hello world"

    def test_punctuation_removed(self, normalizer):
        result = normalizer.process("What's the answer?!", language="en")
        assert "?" not in result.normalized_text
        assert "!" not in result.normalized_text

    def test_french_accents_in_search_text(self, normalizer):
        result = normalizer.process("Les réseaux de neurones", language="fr")
        assert "reseaux" in result.search_text or "réseaux" in result.search_text


class TestEdgeCases:
    def test_empty_text(self, normalizer):
        result = normalizer.process("", language="en")
        assert result.normalized_text == ""
        assert result.original_text == ""

    def test_none_text(self, normalizer):
        result = normalizer.process(None, language="en")
        assert result.normalized_text == ""

    def test_url_removed(self, normalizer):
        result = normalizer.process("Visit https://example.com for more", language="en")
        assert "https" not in result.normalized_text

    def test_word_count(self, normalizer):
        assert normalizer.word_count("one two three") == 3
        assert normalizer.word_count("") == 0
        assert normalizer.word_count(None) == 0
