from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from core.contracts import LanguageDetection


try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except Exception:
    detect_langs = None
    _LANGDETECT_AVAILABLE = False


class LanguageDetector:
    """
    Multilingual language detector optimized for Arabic, English and French.

    Design principles:
    - Strong Unicode heuristics for Arabic script
    - Safe fallback to langdetect when available
    - Stable API:
        detect(text) -> LanguageDetection
        detect_language(text) -> str
        dominant_language(texts) -> LanguageDetection
    """

    _instance: Optional["LanguageDetector"] = None

    ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
    LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
    CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
    CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
    HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

    FRENCH_MARKERS = re.compile(
        r"\b(le|la|les|des|une|un|est|sont|avec|dans|pour|sur|pas|plus|que|qui)\b",
        re.IGNORECASE,
    )
    ENGLISH_MARKERS = re.compile(
        r"\b(the|a|an|is|are|with|in|for|on|not|more|than|what|how|why)\b",
        re.IGNORECASE,
    )
    ARABIC_WORD_MARKERS = re.compile(
        r"\b(هذا|هذه|الذي|التي|في|من|على|ما|كيف|لماذا|هناك|وكان|كانت)\b"
    )

    SUPPORTED = {"ar", "en", "fr", "es", "de", "it", "pt"}

    @classmethod
    def get(cls) -> "LanguageDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str, hint: Optional[str] = None) -> LanguageDetection:
        """
        Detect the language of a single text.
        """
        text = (text or "").strip()
        if not text:
            return LanguageDetection(
                code=hint or "unknown",
                confidence=0.0,
                source="empty",
                script="unknown",
            )

        script = self._dominant_script(text)

        # Strong Arabic heuristic: script first, then lexical confirmation.
        arabic_chars = len(self.ARABIC_RE.findall(text))
        if arabic_chars > 0:
            ratio = arabic_chars / max(1, len(text))
            if ratio >= 0.10 or self.ARABIC_WORD_MARKERS.search(text):
                return LanguageDetection(
                    code="ar",
                    confidence=0.99 if ratio >= 0.20 else 0.93,
                    source="unicode_script",
                    script="arabic",
                )

        # Short text heuristic using lexical markers.
        if len(text) < 48:
            guessed = self._guess_short_text(text, hint=hint, script=script)
            if guessed is not None:
                return guessed

        # Statistical fallback.
        if _LANGDETECT_AVAILABLE and detect_langs is not None:
            try:
                guesses = detect_langs(text)
                if guesses:
                    best = guesses[0]
                    code = self._normalize_lang_code(best.lang, hint=hint)
                    confidence = float(best.prob)
                    return LanguageDetection(
                        code=code,
                        confidence=confidence,
                        source="langdetect",
                        script=script,
                    )
            except Exception:
                pass

        # Fallback on script.
        return LanguageDetection(
            code=self._fallback_from_script(script, hint=hint),
            confidence=0.35,
            source="script_fallback",
            script=script,
        )

    def detect_language(self, text: str, hint: Optional[str] = None) -> str:
        """
        Compatibility helper returning only the language code.
        """
        return self.detect(text=text, hint=hint).code

    def dominant_language(
        self,
        texts: Iterable[str],
        hint: Optional[str] = None,
    ) -> LanguageDetection:
        """
        Detect the dominant language across multiple texts.
        """
        weighted_votes = Counter()
        best_script = "unknown"

        for text in texts:
            det = self.detect(text, hint=hint)
            weighted_votes[det.code] += max(det.confidence, 0.05)
            if best_script == "unknown" and det.script != "unknown":
                best_script = det.script

        if not weighted_votes:
            return LanguageDetection(
                code=hint or "unknown",
                confidence=0.0,
                source="no_texts",
                script="unknown",
            )

        code, score = weighted_votes.most_common(1)[0]
        total = sum(weighted_votes.values()) or 1.0
        return LanguageDetection(
            code=code,
            confidence=float(score) / float(total),
            source="dominant_vote",
            script=best_script,
        )

    def is_rtl(self, language_code: str) -> bool:
        return (language_code or "").lower() in {"ar", "fa", "ur", "he"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_script(self, text: str) -> str:
        counts = {
            "arabic": len(self.ARABIC_RE.findall(text)),
            "latin": len(self.LATIN_RE.findall(text)),
            "cyrillic": len(self.CYRILLIC_RE.findall(text)),
            "cjk": len(self.CJK_RE.findall(text)),
            "hebrew": len(self.HEBREW_RE.findall(text)),
        }
        script, count = max(counts.items(), key=lambda x: x[1])
        return script if count > 0 else "unknown"

    def _guess_short_text(
        self,
        text: str,
        hint: Optional[str],
        script: str,
    ) -> Optional[LanguageDetection]:
        lower = text.lower()

        if self.ARABIC_WORD_MARKERS.search(text):
            return LanguageDetection(
                code="ar",
                confidence=0.95,
                source="short_text_markers",
                script="arabic",
            )

        fr_count = len(self.FRENCH_MARKERS.findall(lower))
        en_count = len(self.ENGLISH_MARKERS.findall(lower))

        if fr_count >= 2 and fr_count > en_count:
            return LanguageDetection(
                code="fr",
                confidence=0.82,
                source="short_text_markers",
                script=script,
            )

        if en_count >= 2 and en_count > fr_count:
            return LanguageDetection(
                code="en",
                confidence=0.82,
                source="short_text_markers",
                script=script,
            )

        if len(text) < 12 and hint:
            return LanguageDetection(
                code=self._normalize_lang_code(hint),
                confidence=0.60,
                source="hint_short_text",
                script=script,
            )

        return None

    def _normalize_lang_code(self, code: str, hint: Optional[str] = None) -> str:
        code = (code or "").strip().lower()

        if code.startswith("ar"):
            return "ar"
        if code.startswith("en"):
            return "en"
        if code.startswith("fr"):
            return "fr"
        if code.startswith("es"):
            return "es"
        if code.startswith("de"):
            return "de"
        if code.startswith("it"):
            return "it"
        if code.startswith("pt"):
            return "pt"

        return hint or "unknown"

    def _fallback_from_script(self, script: str, hint: Optional[str]) -> str:
        if script == "arabic":
            return "ar"
        if script == "latin":
            return hint or "en"
        if script == "hebrew":
            return "he"
        return hint or "unknown"
