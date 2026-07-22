"""
Tests for core.language_detector
"""
import pytest
from core.language_detector import LanguageDetector


@pytest.fixture
def detector():
    return LanguageDetector.get()


class TestArabicDetection:
    def test_arabic_text_detected(self, detector):
        result = detector.detect("السلام عليكم ورحمة الله وبركاته")
        assert result.code == "ar"
        assert result.confidence >= 0.90

    def test_arabic_short_text(self, detector):
        result = detector.detect("مرحبا")
        assert result.code == "ar"

    def test_arabic_with_diacritics(self, detector):
        result = detector.detect("إِنَّ ٱلْعِلْمَ نُورٌ")
        assert result.code == "ar"

    def test_arabic_mixed_with_numbers(self, detector):
        result = detector.detect("الفصل 3: البحث الدلالي في 2024")
        assert result.code == "ar"


class TestEnglishDetection:
    def test_english_text_detected(self, detector):
        result = detector.detect("Today we will discuss retrieval augmented generation.")
        assert result.code == "en"

    def test_english_short(self, detector):
        result = detector.detect("What is deep learning?")
        assert result.code == "en"


class TestFrenchDetection:
    def test_french_text_detected(self, detector):
        result = detector.detect("Aujourd'hui nous allons parler de la recherche sémantique.")
        assert result.code == "fr"

    def test_french_with_accents(self, detector):
        result = detector.detect("Les réseaux de neurones sont très efficaces pour le traitement.")
        assert result.code == "fr"


class TestEdgeCases:
    def test_empty_string(self, detector):
        result = detector.detect("")
        assert result.confidence == 0.0

    def test_none_text(self, detector):
        result = detector.detect(None)
        assert result.confidence == 0.0

    def test_numbers_only(self, detector):
        result = detector.detect("12345 67890")
        # Should not crash, code may be "unknown" or hint-based
        assert isinstance(result.code, str)

    def test_hint_used_for_short_text(self, detector):
        result = detector.detect("ok", hint="fr")
        assert result.code == "fr"


class TestDominantLanguage:
    def test_dominant_from_mixed(self, detector):
        texts = [
            "السلام عليكم",
            "هذا درس مهم",
            "Today is a good day",
            "البحث الدلالي",
        ]
        result = detector.dominant_language(texts)
        assert result.code == "ar"

    def test_dominant_empty(self, detector):
        result = detector.dominant_language([])
        assert result.code in ("unknown", "en")


class TestRTL:
    def test_arabic_is_rtl(self, detector):
        assert detector.is_rtl("ar") is True

    def test_english_is_not_rtl(self, detector):
        assert detector.is_rtl("en") is False

    def test_french_is_not_rtl(self, detector):
        assert detector.is_rtl("fr") is False
