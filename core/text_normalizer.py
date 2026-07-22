from __future__ import annotations

import re
import unicodedata
from typing import Optional

from core.contracts import NormalizedText
from core.language_detector import LanguageDetector


class TextNormalizer:
    """
    Multilingual text normalizer for retrieval.

    Design goals:
    - preserve the original text for display and citation
    - normalize aggressively enough for retrieval
    - support Arabic / English / French well
    - expose compatibility methods used by the rest of the project
    """

    _instance: Optional["TextNormalizer"] = None

    MULTI_SPACE_RE = re.compile(r"\s+")
    GENERIC_PUNCT_RE = re.compile(r"[^\w\s]")
    QUOTE_RE = re.compile(r"[“”«»„‟]")
    DASH_RE = re.compile(r"[-‐‑‒–—―]+")
    URL_RE = re.compile(r"https?://\S+")
    EMAIL_RE = re.compile(r"\S+@\S+")

    ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
    TATWEEL_RE = re.compile(r"\u0640")
    ARABIC_PUNCT_RE = re.compile(r"[،؛؟٪]")
    ARABIC_ALEF_RE = re.compile(r"[إأآٱ]")
    ARABIC_YAA_RE = re.compile(r"[ى]")
    ARABIC_HAMZA_WAW_RE = re.compile(r"[ؤ]")
    ARABIC_HAMZA_YAA_RE = re.compile(r"[ئ]")

    ARABIC_DIGITS_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    @classmethod
    def get(cls) -> "TextNormalizer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str, language: Optional[str] = None) -> NormalizedText:
        """
        Return original / normalized / search text.
        """
        original = (text or "").strip()
        detector = LanguageDetector.get()
        lang = language or detector.detect_language(original)

        if lang == "ar":
            normalized = self._normalize_arabic(original)
            search_text = self._build_arabic_search_text(original, normalized)
        else:
            normalized = self._normalize_latin(original)
            search_text = self._build_latin_search_text(original, normalized)

        return NormalizedText(
            original_text=original,
            normalized_text=normalized,
            search_text=search_text,
            language=lang or "unknown",
        )

    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Compatibility helper returning normalized_text.
        """
        return self.process(text=text, language=language).normalized_text

    def normalize_query(self, text: str, language: Optional[str] = None) -> str:
        """
        Compatibility helper used by retrieval modules.
        """
        return self.process(text=text, language=language).search_text

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        """
        Generic compatibility alias.
        """
        return self.normalize_text(text=text, language=language)

    def word_count(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return 0
        return len(text.split())

    # ------------------------------------------------------------------
    # Internal normalization
    # ------------------------------------------------------------------

    def _normalize_common(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = self.URL_RE.sub(" ", text)
        text = self.EMAIL_RE.sub(" ", text)
        text = self.QUOTE_RE.sub('"', text)
        text = text.replace("’", "'").replace("`", "'")
        text = self.DASH_RE.sub(" ", text)
        text = self.MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    def _normalize_latin(self, text: str) -> str:
        text = self._normalize_common(text)
        text = text.lower()
        text = self.GENERIC_PUNCT_RE.sub(" ", text)
        text = self.MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    def _normalize_arabic(self, text: str) -> str:
        text = self._normalize_common(text)
        text = text.translate(self.ARABIC_DIGITS_MAP)

        # Arabic script normalization
        text = self.ARABIC_DIACRITICS_RE.sub("", text)
        text = self.TATWEEL_RE.sub("", text)
        text = self.ARABIC_ALEF_RE.sub("ا", text)
        text = self.ARABIC_YAA_RE.sub("ي", text)
        text = self.ARABIC_HAMZA_WAW_RE.sub("و", text)
        text = self.ARABIC_HAMZA_YAA_RE.sub("ي", text)

        text = self.ARABIC_PUNCT_RE.sub(" ", text)
        text = self.GENERIC_PUNCT_RE.sub(" ", text)
        text = self.MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    def _remove_accents(self, text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def _build_latin_search_text(self, original: str, normalized: str) -> str:
        accentless = self._remove_accents(normalized)
        variants = [normalized]

        if accentless and accentless != normalized:
            variants.append(accentless)

        compact = normalized.replace("'", " ")
        compact = self.MULTI_SPACE_RE.sub(" ", compact).strip()
        if compact and compact not in variants:
            variants.append(compact)

        # Keep retrieval text compact and deterministic
        deduped = []
        for item in variants:
            if item and item not in deduped:
                deduped.append(item)

        return "\n".join(deduped)

    def _build_arabic_search_text(self, original: str, normalized: str) -> str:
        variants = [normalized]

        # Hamza-softened fallback for imperfect ASR / OCR variants
        softened = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        softened = softened.replace("ة", "ه")
        softened = self.MULTI_SPACE_RE.sub(" ", softened).strip()

        if softened and softened not in variants:
            variants.append(softened)

        deduped = []
        for item in variants:
            if item and item not in deduped:
                deduped.append(item)

        return "\n".join(deduped)
