"""
core/ocr.py — Unified OCR engine v5.0
=======================================
Strategy: one reader per video, chosen from the video's detected language.

    Arabic video  → EasyOCR arabic reader  (ar + en)
    French video  → EasyOCR latin reader   (fr + en)
    English video → EasyOCR latin reader   (en)
    Unknown       → EasyOCR latin reader   (en)  ← safe default

No dual-reader. No script-mixing. One Reader per video, initialized once,
reused for all frames of that video.

Fallback chain:
    EasyOCR → Tesseract → disabled

Public API (unchanged):
    engine = UnifiedOCREngine(languages=["fr", "en"])
    result = engine.extract_text(image_bgr)
    result.text        → extracted text
    result.backend     → "easyocr" | "tesseract" | "none"
    engine.is_available → bool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Safe torch import — needed for gpu=torch.cuda.is_available() in EasyOCR
try:
    import torch as _torch
    def _cuda_available() -> bool:
        return _torch.cuda.is_available()
except ImportError:
    def _cuda_available() -> bool:  # type: ignore[misc]
        return False

# ──────────────────────────────────────────────────────────────────────
# Safe settings
# ──────────────────────────────────────────────────────────────────────

try:
    from config.settings import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any = None) -> Any:
    if _settings is None:
        return default
    return getattr(_settings, name, default)


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════
# Data contracts
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    language_hint: Optional[str] = None


@dataclass
class OCRResult:
    text: str
    lines: List[OCRLine]
    backend: str
    confidence: float


# ══════════════════════════════════════════════════════════════════════
# Language → EasyOCR script mapping
# ══════════════════════════════════════════════════════════════════════

# ISO 639-1 → EasyOCR language list for that script family.
# "en" is always included — EasyOCR needs it as an anchor for most scripts.
LANG_TO_EASYOCR: Dict[str, List[str]] = {
    # Arabic script
    "ar": ["ar", "en"],
    "fa": ["fa", "en"],
    "ur": ["ur", "en"],
    "ug": ["ug", "en"],
    # Latin script
    "fr": ["fr", "en"],
    "en": ["en"],
    "de": ["de", "en"],
    "es": ["es", "en"],
    "it": ["it", "en"],
    "pt": ["pt", "en"],
    "nl": ["nl", "en"],
    "pl": ["pl", "en"],
    "tr": ["tr", "en"],
    # Cyrillic script
    "ru": ["ru", "en"],
    "uk": ["uk", "en"],
    # CJK / other
    "zh": ["ch_sim"],
    "ja": ["ja"],
    "ko": ["ko"],
}

# ISO 639-1 → Tesseract language pack name
TESSERACT_LANG_MAP: Dict[str, str] = {
    "ar": "ara", "en": "eng", "fr": "fra", "de": "deu",
    "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
    "ru": "rus", "fa": "fas", "tr": "tur",
    "zh": "chi_sim", "ja": "jpn", "ko": "kor",
}

CLAHE_SKIP_CONTRAST_RATIO = 4.0


def _easyocr_langs_for(language: str) -> List[str]:
    """
    Return the EasyOCR language list for a given ISO 639-1 code.
    Falls back to ["en"] for unknown codes.
    """
    lang = (language or "en").strip().lower()[:2]
    return LANG_TO_EASYOCR.get(lang, ["en"])


def _tesseract_lang_for(language: str) -> str:
    lang = (language or "en").strip().lower()[:2]
    return TESSERACT_LANG_MAP.get(lang, "eng")


# ══════════════════════════════════════════════════════════════════════
# UnifiedOCREngine v5.0
# ══════════════════════════════════════════════════════════════════════

class UnifiedOCREngine:
    """
    One OCR engine instance per video language.

    Instantiate with the video's detected language:
        engine = UnifiedOCREngine(languages=["fr"])   # French video
        engine = UnifiedOCREngine(languages=["ar"])   # Arabic video

    The first element of `languages` is treated as the primary language
    and determines which EasyOCR model is loaded.
    """

    def __init__(
        self,
        preferred_backend: Optional[str] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ):
        self.preferred_backend = (
            (preferred_backend or _cfg("ocr_backend", "easyocr"))
            .strip().lower()
            .replace("paddleocr", "easyocr")
        )
        # Primary language = first in list
        raw_langs = languages or _cfg("ocr_languages", ["en"])
        self.primary_language: str = (raw_langs[0] if raw_langs else "en").lower()[:2]

        self.min_confidence = float(
            min_confidence if min_confidence is not None
            else _cfg_float("ocr_min_confidence", 0.35)
        )

        self.backend: str = "none"
        self._engine: Any = None
        self._tess_lang: str = ""

        self._init_backend()

    # ==================================================================
    # Public API
    # ==================================================================

    @property
    def is_available(self) -> bool:
        return self.backend != "none"

    def extract_text(self, image_bgr: np.ndarray) -> OCRResult:
        """Extract text from a BGR image (OpenCV format)."""
        if image_bgr is None or image_bgr.size == 0:
            return OCRResult(text="", lines=[], backend=self.backend, confidence=0.0)

        enhanced = self._prepare_image_for_ocr(image_bgr)

        if self.backend == "easyocr":
            return self._extract_easyocr(enhanced)
        if self.backend == "tesseract":
            return self._extract_tesseract(enhanced)

        return OCRResult(text="", lines=[], backend="none", confidence=0.0)

    # ==================================================================
    # Backend init
    # ==================================================================

    def _init_backend(self) -> None:
        if self.preferred_backend == "easyocr" and self._try_init_easyocr():
            return
        if self.preferred_backend == "tesseract" and self._try_init_tesseract():
            return
        if self.preferred_backend != "easyocr" and self._try_init_easyocr():
            return
        if self.preferred_backend != "tesseract" and self._try_init_tesseract():
            return

        logger.warning(
            "⚠️  No OCR backend available. "
            "pip install easyocr  OR  apt install tesseract-ocr && pip install pytesseract"
        )

    def _try_init_easyocr(self) -> bool:
        try:
            import easyocr  # type: ignore
        except ImportError:
            logger.debug("EasyOCR not installed — pip install easyocr")
            return False

        lang_list = _easyocr_langs_for(self.primary_language)
        try:
            logger.info(
                "Initializing EasyOCR for lang=%s → reader langs=%s",
                self.primary_language, lang_list,
            )
            # FORCE CPU: EasyOCR on GPU holds ~1.2 GB resident VRAM that
            # competes with qwen3-vl:8b (6.1 GB). After ~80 frames, VRAM
            # fragments and EasyOCR OOM-crashes on every frame.
            # On CPU: ~80ms/frame for OCR — fast enough, zero VRAM risk.
            self._engine = easyocr.Reader(lang_list, gpu=False, verbose=False)
            logger.info("EasyOCR initialised on CPU (forced — VRAM reserved for VLM)")
            self.backend = "easyocr"
            logger.info(
                "✅ OCR backend: EasyOCR (lang=%s, reader=%s)",
                self.primary_language, lang_list,
            )
            return True
        except Exception as exc:
            logger.warning("EasyOCR init failed (lang=%s): %s", lang_list, exc)
            return False

    def _try_init_tesseract(self) -> bool:
        try:
            import pytesseract  # type: ignore
            pytesseract.get_tesseract_version()
        except ImportError:
            logger.debug("pytesseract not installed — pip install pytesseract")
            return False
        except Exception as exc:
            logger.debug("Tesseract binary not found: %s", exc)
            return False

        try:
            self._tess_lang = _tesseract_lang_for(self.primary_language)
            self._engine = pytesseract
            self.backend = "tesseract"
            logger.info(
                "✅ OCR backend: Tesseract (lang=%s → tess=%s)",
                self.primary_language, self._tess_lang,
            )
            return True
        except Exception as exc:
            logger.warning("Tesseract init failed: %s", exc)
            return False

    # ==================================================================
    # EasyOCR extraction
    # ==================================================================

    def _extract_easyocr(self, image_bgr: np.ndarray) -> OCRResult:
        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            raw = self._engine.readtext(rgb, detail=1, paragraph=False)
        except Exception as exc:
            logger.warning("EasyOCR inference failed: %s", exc)
            return OCRResult(text="", lines=[], backend="easyocr", confidence=0.0)

        lines: List[OCRLine] = []
        for item in raw:
            try:
                bbox = item[0]
                text = str(item[1]).strip()
                confidence = float(item[2])
            except Exception:
                continue

            if not text or confidence < self.min_confidence:
                continue

            lines.append(OCRLine(
                text=text,
                confidence=confidence,
                bbox=[[float(x), float(y)] for x, y in bbox],
                language_hint=self.primary_language,
            ))

        return self._finalize(lines=lines, backend="easyocr")

    # ==================================================================
    # Tesseract extraction
    # ==================================================================

    def _extract_tesseract(self, image_bgr: np.ndarray) -> OCRResult:
        try:
            from pytesseract import Output  # type: ignore
            data = self._engine.image_to_data(
                image_bgr,
                lang=self._tess_lang,
                config="--oem 1 --psm 3",
                output_type=Output.DICT,
            )
        except Exception as exc:
            logger.warning("Tesseract inference failed: %s", exc)
            return OCRResult(text="", lines=[], backend="tesseract", confidence=0.0)

        lines: List[OCRLine] = []
        for i in range(len(data.get("text", []))):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except Exception:
                conf = 0.0
            if conf < self.min_confidence:
                continue

            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            lines.append(OCRLine(
                text=text,
                confidence=conf,
                bbox=[
                    [float(x),     float(y)],
                    [float(x + w), float(y)],
                    [float(x + w), float(y + h)],
                    [float(x),     float(y + h)],
                ],
                language_hint=self.primary_language,
            ))

        return self._finalize(lines=lines, backend="tesseract")

    # ==================================================================
    # Image preprocessing — smart CLAHE
    # ==================================================================

    def _prepare_image_for_ocr(self, image_bgr: np.ndarray) -> np.ndarray:
        image = image_bgr.copy()

        h, w = image.shape[:2]
        if max(h, w) < 1280:
            scale = 1280.0 / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        image = cv2.bilateralFilter(image, 5, 50, 50)

        if self._needs_contrast_enhancement(image):
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            image = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        return image

    @staticmethod
    def _needs_contrast_enhancement(image_bgr: np.ndarray) -> bool:
        try:
            l_ch = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
            _, mask = cv2.threshold(
                l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            bg = l_ch[mask == 255]
            fg = l_ch[mask == 0]
            if len(bg) == 0 or len(fg) == 0:
                return True
            mean_fg = max(float(np.mean(fg)), 1.0)
            return (float(np.mean(bg)) / mean_fg) < CLAHE_SKIP_CONTRAST_RATIO
        except Exception:
            return True

    # ==================================================================
    # Finalize
    # ==================================================================

    def _finalize(self, lines: List[OCRLine], backend: str) -> OCRResult:
        if not lines:
            return OCRResult(text="", lines=[], backend=backend, confidence=0.0)

        lines = sorted(
            lines,
            key=lambda l: (
                min((p[1] for p in l.bbox), default=0.0),
                min((p[0] for p in l.bbox), default=0.0),
            ),
        )
        text = "\n".join(l.text for l in lines if l.text.strip())
        avg_conf = sum(l.confidence for l in lines) / max(1, len(lines))

        return OCRResult(
            text=text.strip(),
            lines=lines,
            backend=backend,
            confidence=float(avg_conf),
        )