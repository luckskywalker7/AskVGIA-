"""
core/chunking.py — Semantic Chunker v4.0
=========================================

NOUVEAUTÉS v4.0 (vs v3.0)
--------------------------

TOPIC-1  Chunking par SUJET / MEANING
    Le chunker ne coupe plus uniquement sur les pauses ou les limites de
    durée. Il détecte les CHANGEMENTS DE SUJET grâce à :
        (a) SemanticSplitDetector (MiniLM) — distance cosine entre phrases
        (b) TopicShiftDetector — marqueurs lexicaux de transition
            ("maintenant", "ensuite", "passons à", "par contre", ...)
            dans 3 langues (AR / EN / FR).
    Les phrases sont toujours complètes : on ne coupe jamais en milieu de
    phrase.

TOPIC-2  Chaque chunk reçoit un TOPIC LABEL généré par le LLM (ou déduit
    par heuristique si le LLM est indisponible).

SUMMARY-1  VideoSummarizer : produit pour chaque vidéo :
        • summary        — résumé global en 3-5 phrases
        • full_text      — transcription complète concaténée
        • topics         — liste des sujets couverts
        • keywords       — mots-clés importants
        • domain         — catégorie principale (voir DOMAIN_TAXONOMY)
        • sub_domain     — sous-catégorie optionnelle

DOMAIN-1  Taxonomie de domaines utilisée pour catégoriser les vidéos :
    Technology, Science, Education, Business, Health, History, Culture,
    Sports, News, Entertainment, Religion, Law, Language, Other.
    La catégorisation est faite par le LLM, avec fallback heuristique.

DOMAIN-2  Le domain + sub_domain sont stockés dans VideoMetadata.extra
    et dans ChunkingResult, permettant la recherche par catégorie.

COMPAT    Toutes les API publiques de v3.0 sont conservées :
    .chunk() / .create_semantic_chunks() / .chunk_segments() /
    .create_chunks() / .process_segments() / .generate_chunks() / .process()

Author : Spiral father, 2026.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Safe settings access
# ──────────────────────────────────────────────────────────────────────

try:
    from config.settings import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any = None) -> Any:
    if _settings is None:
        return default
    return getattr(_settings, name, default)


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(_cfg(name, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(name: str, default: bool) -> bool:
    val = _cfg(name, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


# ──────────────────────────────────────────────────────────────────────
# Safe imports : core.contracts
# ──────────────────────────────────────────────────────────────────────

try:
    from core.contracts import (
        ChunkingResult,
        SemanticChunk,
        TranscriptSegment,
        TranscriptionResult,
    )
except ImportError:
    logger.info("core.contracts not found — using local dataclasses")

    @dataclass
    class TranscriptSegment:
        text: str = ""
        start_time: float = 0.0
        end_time: float = 0.0
        language: str = "unknown"
        speaker: str = "SPEAKER_00"
        confidence: float = 1.0
        words: list = field(default_factory=list)

    @dataclass
    class TranscriptionResult:
        video_id: str = ""
        language: str = "unknown"
        segments: List[TranscriptSegment] = field(default_factory=list)

    @dataclass
    class SemanticChunk:
        chunk_id: str = ""
        video_id: str = ""
        text: str = ""
        original_text: str = ""
        normalized_text: str = ""
        search_text: str = ""
        start_time: float = 0.0
        end_time: float = 0.0
        language: str = "unknown"
        speaker: str = "SPEAKER_00"
        topic: str = "general"
        chunk_type: str = "speech"
        chunk_index: int = 0
        key_points: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ChunkingResult:
        video_id: str = ""
        chunks: List[SemanticChunk] = field(default_factory=list)
        total_segments: int = 0
        total_chunks: int = 0
        language: str = "unknown"
        processing_seconds: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Safe imports : LanguageDetector
# ──────────────────────────────────────────────────────────────────────

_language_detector = None

try:
    from core.language_detector import LanguageDetector
    _language_detector = LanguageDetector.get()
    logger.info("LanguageDetector loaded from core.language_detector")
except ImportError:
    pass
except Exception as exc:
    logger.warning("LanguageDetector init failed: %s", exc)

if _language_detector is None:
    try:
        from lingua import Language, LanguageDetectorBuilder

        class _FallbackLangDetector:
            def __init__(self):
                self._det = (
                    LanguageDetectorBuilder
                    .from_languages(Language.ARABIC, Language.ENGLISH, Language.FRENCH)
                    .with_minimum_relative_distance(0.25)
                    .build()
                )
                self._map = {
                    Language.ARABIC: "ar",
                    Language.ENGLISH: "en",
                    Language.FRENCH: "fr",
                }

            def detect_language(self, text: str, hint: str = "en") -> str:
                if not text or not text.strip():
                    return hint
                try:
                    r = self._det.detect_language_of(text)
                    return self._map.get(r, hint)
                except Exception:
                    return hint

            def dominant_language(self, texts: List[str]):
                from collections import Counter
                langs = [self.detect_language(t) for t in texts if t.strip()]
                if not langs:
                    return type("L", (), {"code": "en"})()
                most = Counter(langs).most_common(1)[0][0]
                return type("L", (), {"code": most})()

            @classmethod
            def get(cls):
                return cls()

        _language_detector = _FallbackLangDetector()
        logger.info("Fallback lingua LanguageDetector loaded")
    except ImportError:
        pass

if _language_detector is None:
    class _HeuristicLangDetector:
        def detect_language(self, text: str, hint: str = "en") -> str:
            if not text:
                return hint
            arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
            if arabic > len(text) * 0.2:
                return "ar"
            accented = sum(1 for c in text if c in "àâäéèêëïîôùûüçœæ")
            if accented >= 2:
                return "fr"
            return hint

        def dominant_language(self, texts: List[str]):
            from collections import Counter
            langs = [self.detect_language(t) for t in texts if t.strip()]
            if not langs:
                return type("L", (), {"code": "en"})()
            most = Counter(langs).most_common(1)[0][0]
            return type("L", (), {"code": most})()

        @classmethod
        def get(cls):
            return cls()

    _language_detector = _HeuristicLangDetector()
    logger.info("Heuristic LanguageDetector loaded")


# ──────────────────────────────────────────────────────────────────────
# Safe imports : TextNormalizer
# ──────────────────────────────────────────────────────────────────────

_text_normalizer = None

try:
    from core.text_normalizer import TextNormalizer
    _text_normalizer = TextNormalizer.get()
    logger.info("TextNormalizer loaded from core.text_normalizer")
except ImportError:
    pass
except Exception as exc:
    logger.warning("TextNormalizer init failed: %s", exc)

if _text_normalizer is None:
    class _FallbackNormalizer:
        def process(self, text: str, language: str = "en"):
            cleaned = " ".join(text.split()).strip()
            return type("N", (), {
                "normalized_text": cleaned,
                "search_text": cleaned,
            })()

        @classmethod
        def get(cls):
            return cls()

    _text_normalizer = _FallbackNormalizer()
    logger.info("Fallback TextNormalizer loaded")


# ──────────────────────────────────────────────────────────────────────
# numpy
# ──────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


# ══════════════════════════════════════════════════════════════════════
# DOMAIN TAXONOMY
# ══════════════════════════════════════════════════════════════════════

#: Canonical domain labels used for video categorization.
DOMAIN_TAXONOMY: Dict[str, List[str]] = {
    "Technology":    ["software", "programming", "AI", "machine learning", "hardware",
                      "cybersecurity", "internet", "robotics", "data", "algorithm",
                      "informatique", "logiciel", "تقنية", "برمجة", "ذكاء اصطناعي"],
    "Science":       ["physics", "chemistry", "biology", "mathematics", "astronomy",
                      "research", "experiment", "theory", "quantum", "genome",
                      "science", "physique", "chimie", "علم", "فيزياء", "كيمياء"],
    "Education":     ["learning", "teaching", "school", "university", "course",
                      "lecture", "lesson", "student", "education", "tutorial",
                      "apprentissage", "cours", "تعليم", "درس", "مدرسة"],
    "Business":      ["finance", "economy", "startup", "marketing", "investment",
                      "management", "entrepreneur", "strategy", "market", "revenue",
                      "économie", "entreprise", "اقتصاد", "تجارة", "استثمار"],
    "Health":        ["medicine", "health", "doctor", "disease", "treatment",
                      "nutrition", "fitness", "mental health", "hospital", "therapy",
                      "santé", "médecine", "صحة", "طب", "علاج"],
    "History":       ["history", "war", "civilization", "ancient", "empire",
                      "revolution", "historical", "century", "dynasty", "heritage",
                      "histoire", "guerre", "تاريخ", "حضارة", "حرب"],
    "Culture":       ["art", "music", "film", "literature", "culture", "philosophy",
                      "religion", "tradition", "society", "identity",
                      "culture", "art", "ثقافة", "فن", "أدب"],
    "Sports":        ["sport", "football", "basketball", "athlete", "competition",
                      "match", "team", "championship", "training", "player",
                      "sport", "foot", "رياضة", "كرة", "بطولة"],
    "News":          ["news", "politics", "government", "election", "policy",
                      "president", "minister", "parliament", "crisis", "conflict",
                      "actualité", "politique", "أخبار", "سياسة", "انتخابات"],
    "Entertainment": ["comedy", "entertainment", "game", "show", "celebrity",
                      "interview", "podcast", "vlog", "review", "fun",
                      "divertissement", "jeu", "ترفيه", "كوميديا", "مقابلة"],
    "Religion":      ["islam", "quran", "prayer", "religion", "faith", "god",
                      "christian", "bible", "spiritual", "mosque",
                      "religion", "foi", "دين", "إسلام", "قرآن", "صلاة"],
    "Law":           ["law", "legal", "court", "justice", "rights", "constitution",
                      "lawyer", "judge", "regulation", "contract",
                      "droit", "loi", "قانون", "محكمة", "عدالة"],
    "Language":      ["language", "grammar", "vocabulary", "pronunciation", "arabic",
                      "english", "french", "translation", "linguistics", "dialect",
                      "langue", "grammaire", "لغة", "نحو", "صرف"],
    "Other":         [],
}


def detect_domain_heuristic(text: str, keywords: List[str]) -> Tuple[str, str]:
    """
    Heuristic domain detection based on keyword frequency.
    Returns (domain, sub_domain).
    """
    combined = (text + " " + " ".join(keywords)).lower()
    scores: Dict[str, int] = {}
    for domain, markers in DOMAIN_TAXONOMY.items():
        if domain == "Other":
            continue
        count = sum(1 for m in markers if m.lower() in combined)
        if count > 0:
            scores[domain] = count

    if not scores:
        return "Other", ""

    top = sorted(scores.items(), key=lambda x: -x[1])
    domain = top[0][0]
    sub_domain = top[1][0] if len(top) > 1 else ""
    return domain, sub_domain


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

SPEAKER_GRACE_PERIOD_S = 0.8
MULTILINGUAL_THRESHOLD = 0.15
MAX_MERGE_GAP_S = 10.0

GAP_THRESHOLDS: Dict[str, float] = {
    "ar": 2.5,
    "fa": 2.5,
    "ur": 2.5,
    "fr": 1.2,
    "en": 1.5,
    "es": 1.3,
    "de": 1.4,
    "it": 1.3,
}
DEFAULT_GAP_THRESHOLD = 1.5


# ─────────────────────────────────────────────────────────────────────
# ADAPTIVE CHUNK PROFILE  (fix for "retrieved but diluted")
# Estimate, BEFORE packing, whether the video is a dense lecture or
# conversational content, and pick a chunk size/overlap accordingly.
# Model-free: speech rate + multilingual keyword densities.
# Reversible via config flag `chunking_adaptive_profile_enabled`.
# ─────────────────────────────────────────────────────────────────────

_CHUNK_PROFILES: Dict[str, Dict[str, float]] = {
    "lecture": {
        "max_chunk_words": 140,
        "max_chunk_seconds": 24.0,
        "overlap_seconds": 6.0,
        "semantic_sensitivity": 0.4,
    },
    "conversational": {
        "max_chunk_words": 220,
        "max_chunk_seconds": 35.0,
        "overlap_seconds": 3.0,
        "semantic_sensitivity": 0.5,
    },
}

_LECTURE_KEYWORDS = re.compile(
    r"\b(therefore|thus|hence|as we can see|in this (?:lecture|tutorial|video|lesson)"
    r"|let us|let's define|the formula|equation|theorem|chapter|in summary|to summari[sz]e"
    r"|first(?:ly)?|second(?:ly)?|finally|for example|note that|recall that"
    r"|donc|ainsi|par conséquent|comme on peut le voir|la formule|le théorème|chapitre"
    r"|premièrement|deuxièmement|enfin|par exemple|rappelons"
    r"|لذلك|إذن|كما نرى|المعادلة|النظرية|الفصل|أولا|ثانيا|أخيرا|على سبيل المثال)\b",
    re.IGNORECASE,
)
_CASUAL_KEYWORDS = re.compile(
    r"\b(you know|i mean|like,|yeah|kind of|sort of|anyway|so,|right\?|honestly"
    r"|gonna|wanna|stuff|guys|um|uh"
    r"|tu sais|je veux dire|ouais|genre|bref|enfin bref|d'accord|quoi"
    r"|يعني|تعرف|نوعا ما|على كل حال|طيب)\b",
    re.IGNORECASE,
)


def estimate_chunk_profile(segments: List[Any]) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Decide a chunk profile from RAW segments, before packing. Model-free.
    Returns (profile_name, profile_dict, debug_metrics).
    """
    if not segments:
        return "conversational", dict(_CHUNK_PROFILES["conversational"]), {}

    texts: List[str] = []
    total_words = 0
    first_start: Optional[float] = None
    last_end = 0.0
    speech_seconds = 0.0

    for seg in segments:
        t = _seg_get(seg, "text", "") or ""
        texts.append(t)
        total_words += len(t.split())
        s = _safe_float(_seg_get(seg, "start_time", 0.0))
        e = _safe_float(_seg_get(seg, "end_time", s))
        if first_start is None:
            first_start = s
        last_end = max(last_end, e)
        if e > s:
            speech_seconds += (e - s)

    full_text = " ".join(texts)
    span_seconds = max((last_end - (first_start or 0.0)), 1.0)
    denom = speech_seconds if speech_seconds > 5.0 else span_seconds
    speech_rate = total_words / denom if denom > 0 else 0.0

    per_k = (total_words / 1000.0) or 1.0
    lecture_density = len(_LECTURE_KEYWORDS.findall(full_text)) / per_k
    casual_density = len(_CASUAL_KEYWORDS.findall(full_text)) / per_k

    lecture_score = 0.0
    if speech_rate >= 2.5:
        lecture_score += 1.0
    elif speech_rate >= 2.1:
        lecture_score += 0.5
    lecture_score += min(lecture_density * 0.5, 1.0)
    lecture_score -= min(casual_density * 0.6, 1.5)

    profile_name = "lecture" if lecture_score >= 0.8 else "conversational"
    metrics = {
        "speech_rate_wps": round(speech_rate, 2),
        "total_words": float(total_words),
        "lecture_density_per_1k": round(lecture_density, 2),
        "casual_density_per_1k": round(casual_density, 2),
        "lecture_score": round(lecture_score, 2),
    }
    return profile_name, dict(_CHUNK_PROFILES[profile_name]), metrics


# ── Topic-shift lexical markers (AR / EN / FR) ────────────────────────
# These words signal that the speaker is changing subject.
TOPIC_SHIFT_MARKERS: Dict[str, List[str]] = {
    "en": [
        "now", "next", "moving on", "let's talk about", "let me explain",
        "another topic", "on the other hand", "however", "in contrast",
        "furthermore", "additionally", "in conclusion", "to summarize",
        "firstly", "secondly", "thirdly", "finally", "first of all",
        "turning to", "switching to", "let's now", "what about",
    ],
    "fr": [
        "maintenant", "ensuite", "passons à", "parlons de", "d'un autre côté",
        "cependant", "en revanche", "de plus", "par ailleurs", "pour conclure",
        "en résumé", "premièrement", "deuxièmement", "enfin", "tout d'abord",
        "abordons", "venons-en à", "concernant", "quant à", "en ce qui concerne",
    ],
    "ar": [
        "الآن", "بعد ذلك", "دعنا نتحدث عن", "من ناحية أخرى",
        "ومع ذلك", "علاوة على ذلك", "في الختام", "باختصار",
        "أولاً", "ثانياً", "ثالثاً", "أخيراً", "وبالإضافة إلى ذلك",
        "أما بالنسبة", "فيما يخص", "وننتقل الآن", "وعلى صعيد آخر",
        "ولننتقل إلى", "في سياق آخر",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _chunk_id(video_id: str, start: float, end: float, text: str) -> str:
    raw = f"{video_id}|{start:.3f}|{end:.3f}|{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _word_count(text: str) -> int:
    text = (text or "").strip()
    return len(text.split()) if text else 0


def _seg_get(seg: Any, name: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _is_sentence_end(text: str, language: str = "en") -> bool:
    """Language-aware sentence boundary detection."""
    text = (text or "").strip()
    if not text:
        return False
    last_char = text[-1]
    if last_char in (".", "!", "?", "…"):
        return True
    if language in ("ar", "fa", "ur"):
        if last_char in ("\u061F", "\u061B", "\u060C"):
            return True
        arabic_finals = (
            "\u0623\u0646", "\u0625\u0646", "\u0644\u0643\u0646",
            "\u062D\u064A\u062B", "\u0644\u0630\u0644\u0643",
            "\u0647\u0643\u0630\u0627",
        )
        for particle in arabic_finals:
            if text.endswith(particle):
                return True
    if language == "fr":
        if last_char in ("\u00BB", ":", ";"):
            return True
    if last_char in (":", ";"):
        return True
    return False


def _gap_threshold(language: str, fallback: float = DEFAULT_GAP_THRESHOLD) -> float:
    return GAP_THRESHOLDS.get(language, fallback)


def _language_switched(seg_a: Any, seg_b: Any) -> bool:
    lang_a = (_seg_get(seg_a, "language") or "").strip().lower()
    lang_b = (_seg_get(seg_b, "language") or "").strip().lower()
    if not lang_a or not lang_b:
        return False
    if lang_a == "unknown" or lang_b == "unknown":
        return False
    return lang_a != lang_b


def _speaker_changed_meaningfully(seg_a: Any, seg_b: Any) -> bool:
    spk_a = _seg_get(seg_a, "speaker")
    spk_b = _seg_get(seg_b, "speaker")
    if not spk_a or not spk_b:
        return False
    if spk_a == spk_b:
        return False
    start_a = _safe_float(_seg_get(seg_a, "start_time"))
    end_a = _safe_float(_seg_get(seg_a, "end_time"))
    start_b = _safe_float(_seg_get(seg_b, "start_time"))
    seg_a_duration = max(0.0, end_a - start_a)
    if seg_a_duration < SPEAKER_GRACE_PERIOD_S:
        return False
    gap = max(0.0, start_b - end_a)
    if gap < 0.1 and seg_a_duration < 1.0:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════
# TOPIC-1 : TopicShiftDetector — lexical markers
# ══════════════════════════════════════════════════════════════════════

class TopicShiftDetector:
    """
    Detects topic-change signals via lexical transition markers.
    Works for Arabic, English, French.

    Returns True if the beginning of `text` contains a strong
    topic-shift marker (e.g. "now let's talk about", "passons à", "الآن").
    """

    def __init__(self):
        # Pre-compile patterns for speed
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for lang, markers in TOPIC_SHIFT_MARKERS.items():
            self._patterns[lang] = [
                re.compile(
                    r"(^|\s)" + re.escape(m.lower()),
                    re.IGNORECASE | re.UNICODE,
                )
                for m in markers
            ]

    def is_topic_shift(self, text: str, language: str = "en") -> bool:
        """
        Check whether `text` starts with a topic-change signal.
        Only checks the first 120 characters to avoid false positives.
        """
        if not text:
            return False
        head = text.strip()[:120].lower()
        lang = language[:2].lower() if language else "en"
        patterns = self._patterns.get(lang, self._patterns.get("en", []))
        for pat in patterns:
            if pat.search(head):
                return True
        return False

    def score(self, text: str, language: str = "en") -> float:
        """Return 0.0 (no shift) or 1.0 (shift detected)."""
        return 1.0 if self.is_topic_shift(text, language) else 0.0


# ══════════════════════════════════════════════════════════════════════
# SemanticSplitDetector — MiniLM cosine distances
# ══════════════════════════════════════════════════════════════════════

class SemanticSplitDetector:
    """
    Uses multilingual MiniLM to compute semantic distances between
    sentence units. Adaptive threshold: mean + 0.5 × std.
    """

    def __init__(self):
        self.model = None
        self._available = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Use local model instead of downloading
            import pathlib as _pathlib
            _project_root = _pathlib.Path(__file__).resolve().parent.parent
            _model_path = _project_root / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            self.model = SentenceTransformer(str(_model_path))
            self._available = True
            logger.info("SemanticSplitDetector loaded (MiniLM multilingual) from local path")
        except ImportError:
            logger.debug("sentence-transformers not installed, semantic splitting disabled")
        except Exception as exc:
            logger.warning("SemanticSplitDetector load failed: %s", exc, exc_info=True)
    @property
    def available(self) -> bool:
        return self._available

    def compute_distances(self, texts: List[str]) -> List[float]:
        if not self._available or not self.model or len(texts) < 2:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            if _NP:
                emb = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                normed = emb / norms
                sims = np.sum(normed[:-1] * normed[1:], axis=1)
                distances = (1.0 - sims).tolist()
            else:
                distances = []
                for i in range(len(embeddings) - 1):
                    a = embeddings[i]
                    b = embeddings[i + 1]
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b))
                    if norm_a > 0 and norm_b > 0:
                        distances.append(1.0 - dot / (norm_a * norm_b))
                    else:
                        distances.append(1.0)
            return distances
        except Exception as exc:
            logger.warning("Semantic distance computation failed: %s", exc)
            return []

    def find_split_points(
        self,
        texts: List[str],
        sensitivity: float = 0.5,
    ) -> List[int]:
        """
        Returns indices i where a topic change is detected between
        texts[i] and texts[i+1].

        sensitivity : multiplier on std (lower = more splits).
        """
        distances = self.compute_distances(texts)
        if not distances:
            return []
        mean_d = sum(distances) / len(distances)
        variance = sum((d - mean_d) ** 2 for d in distances) / len(distances)
        std_d = math.sqrt(variance)
        threshold = mean_d + sensitivity * std_d
        return [i for i, d in enumerate(distances) if d > threshold]


# ══════════════════════════════════════════════════════════════════════
# TOPIC-2 : TopicLabeler — LLM or heuristic topic label per chunk
# ══════════════════════════════════════════════════════════════════════

class TopicLabeler:
    """
    Assigns a short topic label to each chunk group.

    Strategy:
        1. If LLM (chat_json) is available → ask for a 3-5 word label.
        2. Otherwise → extract the most frequent non-stop nouns from the text.
    """

    # Multilingual stop words (minimal)
    _STOP: Set[str] = {
        # EN
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "to", "of", "in", "on",
        "at", "by", "for", "with", "as", "and", "or", "but", "not",
        "this", "that", "it", "its", "i", "we", "you", "he", "she",
        "they", "so", "if", "from", "up", "about", "there", "what",
        "when", "how", "which", "who", "all", "just", "also", "more",
        "than", "very", "no", "my", "your", "his", "her", "their",
        "like", "can", "let", "now", "then", "here", "these", "those",
        # FR
        "le", "la", "les", "un", "une", "des", "du", "est", "sont",
        "avec", "dans", "pour", "sur", "pas", "plus", "que", "qui",
        "ce", "se", "je", "nous", "vous", "ils", "elles", "leur",
        # AR
        "في", "من", "على", "إلى", "مع", "هذا", "هذه", "كان", "كانت",
        "التي", "الذي", "هو", "هي", "عن", "أن", "إن", "لا", "ما",
    }

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm_available: Optional[bool] = None

    def _try_llm(self, text: str, language: str) -> Optional[str]:
        """Try to get a topic label from the LLM (chat_json)."""
        try:
            from core.llm_client import chat_json  # type: ignore

            lang_name = {"ar": "Arabic", "fr": "French", "en": "English"}.get(
                language[:2].lower(), "English"
            )
            prompt = (
                f"You are a topic labeler. Read the following text excerpt and produce "
                f"a SHORT topic label (3-5 words max, in {lang_name}).\n"
                f"Rules:\n"
                f"- The label must capture the MAIN SUBJECT of the text.\n"
                f"- Output ONLY a JSON object: {{\"topic\": \"...\"}}\n"
                f"- No explanation, no markdown.\n\n"
                f"Text:\n{text[:600]}"
            )
            result = chat_json(
                prompt,
                system="You are a precise topic labeler. Output only JSON.",
                num_predict=40,
                temperature=0.1,
            )
            if isinstance(result, dict):
                label = str(result.get("topic", "") or "").strip()
                if label and len(label.split()) <= 8:
                    self._llm_available = True
                    return label
        except Exception as exc:
            logger.debug("TopicLabeler LLM call failed: %s", exc)
            self._llm_available = False
        return None

    def _heuristic_label(self, text: str, language: str) -> str:
        """Extract the 2-3 most frequent meaningful words as a topic label."""
        from collections import Counter
        words = re.findall(r"\b[\w\u0600-\u06FF]{4,}\b", text.lower())
        words = [w for w in words if w not in self._STOP]
        if not words:
            return "general"
        counts = Counter(words).most_common(3)
        return " ".join(w for w, _ in counts)

    def label(self, text: str, language: str = "en") -> str:
        """Return a topic label string for the given text."""
        if not text or not text.strip():
            return "general"

        # Try LLM if not known to be unavailable
        if self.use_llm and self._llm_available is not False:
            label = self._try_llm(text, language)
            if label:
                return label

        return self._heuristic_label(text, language)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY-1 : VideoSummarizer
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VideoSummary:
    """
    Full video-level summary produced by VideoSummarizer.

    Stored in ChunkingResult.metadata and in VideoMetadata.extra.
    """
    video_id: str
    full_text: str                          # Complete concatenated transcript
    summary: str                            # 3-5 sentence overview
    topics: List[str] = field(default_factory=list)     # subjects covered
    keywords: List[str] = field(default_factory=list)   # key terms
    domain: str = "Other"                   # from DOMAIN_TAXONOMY
    sub_domain: str = ""                    # secondary domain
    language: str = "unknown"
    duration_seconds: float = 0.0
    total_chunks: int = 0
    chunk_topics: List[str] = field(default_factory=list)  # per-chunk topic list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "full_text": self.full_text,
            "summary": self.summary,
            "topics": self.topics,
            "keywords": self.keywords,
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "total_chunks": self.total_chunks,
            "chunk_topics": self.chunk_topics,
        }


_SUMMARY_PROMPT_TEMPLATE = """\
You are an expert video content analyst. Below is the FULL TRANSCRIPT of a video.

Produce a JSON object with EXACTLY these fields and nothing else:
{{
  "summary": "3 to 5 sentence overview of what this video is about",
  "topics": ["list of 3 to 8 specific subjects discussed in this video"],
  "keywords": ["8 to 15 important terms or phrases from the video"],
  "domain": "ONE category from this list: Technology, Science, Education, Business, Health, History, Culture, Sports, News, Entertainment, Religion, Law, Language, Other",
  "sub_domain": "optional secondary category from the same list, or empty string"
}}

Rules:
- Write the summary and topics in the SAME language as the transcript.
- Keywords should be specific and meaningful (no stop words).
- Domain must be EXACTLY one of the listed categories.
- Output ONLY the JSON object. No markdown, no explanation.

Transcript ({language}, {duration}):
---
{transcript}
---
"""


class VideoSummarizer:
    """
    Generates a VideoSummary from a list of SemanticChunks.

    Calls the LLM once with the full transcript (truncated to ~3000 chars
    if too long). Falls back to heuristics if LLM is unavailable.
    """

    def __init__(self, use_llm: bool = True, max_transcript_chars: int = 3000):
        self.use_llm = use_llm
        self.max_transcript_chars = max_transcript_chars

    def summarize(
        self,
        video_id: str,
        chunks: List[SemanticChunk],
        language: str = "unknown",
        duration_seconds: float = 0.0,
    ) -> VideoSummary:
        """
        Build a VideoSummary from the chunked transcript.
        """
        if not chunks:
            return VideoSummary(
                video_id=video_id,
                full_text="",
                summary="No content available.",
                language=language,
                duration_seconds=duration_seconds,
            )

        # ── Build full transcript ────────────────────────────────────
        text_parts: List[str] = []
        for c in chunks:
            t = (
                getattr(c, "text", "")
                or getattr(c, "original_text", "")
                or (c.get("text") if isinstance(c, dict) else "")
                or ""
            ).strip()
            if t:
                text_parts.append(t)

        full_text = " ".join(text_parts)
        chunk_topics = [
            str(getattr(c, "topic", "") or (c.get("topic") if isinstance(c, dict) else "") or "general")
            for c in chunks
        ]

        # ── Try LLM ─────────────────────────────────────────────────
        parsed: Optional[Dict[str, Any]] = None
        if self.use_llm:
            parsed = self._call_llm(full_text, language, duration_seconds)

        # ── Build VideoSummary ───────────────────────────────────────
        if parsed:
            domain = str(parsed.get("domain", "Other") or "Other").strip()
            if domain not in DOMAIN_TAXONOMY:
                domain = "Other"
            sub_domain = str(parsed.get("sub_domain", "") or "").strip()
            if sub_domain and sub_domain not in DOMAIN_TAXONOMY:
                sub_domain = ""

            return VideoSummary(
                video_id=video_id,
                full_text=full_text,
                summary=str(parsed.get("summary", "") or "").strip(),
                topics=[str(t) for t in parsed.get("topics", []) if t][:8],
                keywords=[str(k) for k in parsed.get("keywords", []) if k][:15],
                domain=domain,
                sub_domain=sub_domain,
                language=language,
                duration_seconds=duration_seconds,
                total_chunks=len(chunks),
                chunk_topics=chunk_topics,
            )

        # ── Heuristic fallback ───────────────────────────────────────
        logger.warning("VideoSummarizer: LLM unavailable for %s, using heuristics", video_id)
        return self._heuristic_summary(
            video_id, full_text, chunks, language, duration_seconds, chunk_topics
        )

    def _call_llm(
        self,
        full_text: str,
        language: str,
        duration_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM (chat_json) for structured summary."""
        try:
            from core.llm_client import chat_json  # type: ignore
        except ImportError:
            return None

        # Truncate if too long
        transcript = full_text
        if len(transcript) > self.max_transcript_chars:
            # Take beginning + end for better coverage
            half = self.max_transcript_chars // 2
            transcript = transcript[:half] + "\n[...]\n" + transcript[-half:]

        duration_str = _format_duration(duration_seconds)
        lang_name = {"ar": "Arabic", "fr": "French", "en": "English"}.get(
            (language or "en")[:2].lower(), "English"
        )

        prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            language=lang_name,
            duration=duration_str,
            transcript=transcript,
        )

        try:
            result = chat_json(
                prompt,
                system=(
                    "You are a precise video analyst. "
                    "Output only a valid JSON object as instructed."
                ),
                num_predict=512,
                temperature=0.1,
            )
            if isinstance(result, dict) and result.get("summary"):
                return result
        except Exception as exc:
            logger.warning("VideoSummarizer LLM call failed: %s", exc)

        return None

    def _heuristic_summary(
        self,
        video_id: str,
        full_text: str,
        chunks: List[Any],
        language: str,
        duration_seconds: float,
        chunk_topics: List[str],
    ) -> VideoSummary:
        """Build a summary using frequency-based heuristics."""
        from collections import Counter

        # Summary = first 3 chunks text
        summary_parts = []
        for c in chunks[:3]:
            t = (
                getattr(c, "text", "")
                or (c.get("text") if isinstance(c, dict) else "")
                or ""
            ).strip()
            if t:
                summary_parts.append(t[:200])
        summary = " ".join(summary_parts)[:600] or "No summary available."

        # Keywords = most frequent meaningful words
        STOP = {
            "the", "a", "an", "is", "are", "was", "be", "have", "do",
            "to", "of", "in", "on", "at", "by", "for", "with", "and",
            "or", "but", "not", "this", "that", "it", "we", "you",
            "le", "la", "les", "des", "un", "une", "est", "dans",
            "في", "من", "على", "إلى", "مع", "هذا", "التي", "الذي",
        }
        words = re.findall(r"\b[\w\u0600-\u06FF]{4,}\b", full_text.lower())
        counter = Counter(w for w in words if w not in STOP)
        keywords = [w for w, _ in counter.most_common(12)]

        # Topics = unique chunk topics (deduped, max 6)
        seen: Set[str] = set()
        topics: List[str] = []
        for t in chunk_topics:
            if t and t not in seen and t != "general":
                seen.add(t)
                topics.append(t)
            if len(topics) >= 6:
                break

        # Domain detection
        domain, sub_domain = detect_domain_heuristic(full_text[:2000], keywords)

        return VideoSummary(
            video_id=video_id,
            full_text=full_text,
            summary=summary,
            topics=topics,
            keywords=keywords,
            domain=domain,
            sub_domain=sub_domain,
            language=language,
            duration_seconds=duration_seconds,
            total_chunks=len(chunks),
            chunk_topics=chunk_topics,
        )


def _format_duration(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "unknown duration"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ══════════════════════════════════════════════════════════════════════
# SemanticChunker v4.0
# ══════════════════════════════════════════════════════════════════════

class SemanticChunker:
    """
    Multilingual semantic chunker v4.0.

    Chunking strategy (in order of priority):
    ──────────────────────────────────────────
    1. TOPIC SHIFT (lexical markers)   → always causes a chunk boundary
    2. SEMANTIC SHIFT (MiniLM cosine)  → causes boundary when distance > threshold
    3. SENTENCE BOUNDARY               → never cuts inside a sentence
    4. SPEAKER CHANGE                  → optional (split_on_speaker_change)
    5. LANGUAGE SWITCH                 → always causes a boundary
    6. GAP / PAUSE                     → language-aware threshold
    7. HARD LIMITS                     → max_chunk_words / max_chunk_seconds

    Post-chunking:
    ──────────────
    - Each chunk receives a TOPIC LABEL (LLM or heuristic).
    - A VideoSummary is built once for the whole video.
    - domain + sub_domain are stored in metadata.
    """

    def __init__(
        self,
        overlap_seconds: Optional[float] = None,
        max_chunk_words: Optional[int] = None,
        min_chunk_words: Optional[int] = None,
        max_chunk_seconds: Optional[float] = None,
        min_chunk_seconds: Optional[float] = None,
        split_on_speaker_change: Optional[bool] = None,
        merge_short_segments: Optional[bool] = None,
        max_gap_seconds: Optional[float] = None,
        semantic_enabled: Optional[bool] = None,
        topic_labeling_enabled: Optional[bool] = None,
        summarization_enabled: Optional[bool] = None,
        semantic_sensitivity: Optional[float] = None,
        **_: Any,
    ):
        self.overlap_seconds = float(
            overlap_seconds if overlap_seconds is not None
            else _cfg_float("chunking_overlap_seconds", 3.0)
        )
        self.max_chunk_words = int(
            max_chunk_words if max_chunk_words is not None
            else _cfg_int("chunking_max_words", 220)
        )
        self.min_chunk_words = int(
            min_chunk_words if min_chunk_words is not None
            else _cfg_int("chunking_min_words", 15)
        )
        self.max_chunk_seconds = float(
            max_chunk_seconds if max_chunk_seconds is not None
            else _cfg_float("chunking_max_duration_seconds", 35.0)
        )
        self.min_chunk_seconds = float(
            min_chunk_seconds if min_chunk_seconds is not None
            else _cfg_float("chunking_min_duration_seconds", 2.0)
        )
        self.split_on_speaker_change = bool(
            split_on_speaker_change if split_on_speaker_change is not None
            else _cfg_bool("chunking_split_on_speaker_change", True)
        )
        self.merge_short_segments = bool(
            merge_short_segments if merge_short_segments is not None
            else _cfg_bool("chunking_merge_short_segments", True)
        )
        self.max_gap_seconds = float(
            max_gap_seconds if max_gap_seconds is not None
            else _cfg_float("chunking_max_gap_seconds", 1.5)
        )
        self.semantic_enabled = bool(
            semantic_enabled if semantic_enabled is not None
            else _cfg_bool("chunking_semantic_enabled", True)
        )
        self.topic_labeling_enabled = bool(
            topic_labeling_enabled if topic_labeling_enabled is not None
            else _cfg_bool("chunking_topic_labeling_enabled", True)
        )
        self.summarization_enabled = bool(
            summarization_enabled if summarization_enabled is not None
            else _cfg_bool("chunking_summarization_enabled", True)
        )
        # How sensitive is semantic splitting (lower = more splits)
        self.semantic_sensitivity = float(
            semantic_sensitivity if semantic_sensitivity is not None
            else _cfg_float("chunking_semantic_sensitivity", 0.5)
        )

        # Adaptive per-video chunk profile (lecture vs conversational).
        self.adaptive_profile_enabled = bool(
            _cfg_bool("chunking_adaptive_profile_enabled", True)
        )
        # Remember the configured defaults so each video starts from them
        # before adaptive tuning (otherwise profiles would compound video-to-video).
        self._default_max_chunk_words = self.max_chunk_words
        self._default_max_chunk_seconds = self.max_chunk_seconds
        self._default_overlap_seconds = self.overlap_seconds
        self._default_semantic_sensitivity = self.semantic_sensitivity

        # Core tools
        self.detector = _language_detector
        self.normalizer = _text_normalizer
        self._topic_shift_detector = TopicShiftDetector()
        self._topic_labeler = TopicLabeler(use_llm=self.topic_labeling_enabled)
        self._video_summarizer = VideoSummarizer(use_llm=self.summarization_enabled)

        # Optional semantic splitter
        self._semantic_detector: Optional[SemanticSplitDetector] = None
        if self.semantic_enabled:
            self._semantic_detector = SemanticSplitDetector()
            if not self._semantic_detector.available:
                self._semantic_detector = None

        logger.info(
            "SemanticChunker v4.0 | max_words=%d max_sec=%.1f overlap=%.1f "
            "semantic=%s topic_labels=%s summary=%s sensitivity=%.2f",
            self.max_chunk_words, self.max_chunk_seconds, self.overlap_seconds,
            "ON" if self._semantic_detector else "OFF",
            "ON" if self.topic_labeling_enabled else "OFF",
            "ON" if self.summarization_enabled else "OFF",
            self.semantic_sensitivity,
        )

    # ==================================================================
    # Public API (all v3.0 aliases preserved)
    # ==================================================================

    def _apply_adaptive_profile(self, segments: List[Any]) -> None:
        """
        Estimate the chunk profile for THIS video and set per-video size/overlap.
        Always resets to configured defaults first so profiles don't compound
        across videos. Never raises — falls back to defaults on any error.
        """
        # reset to defaults for every video
        self.max_chunk_words = self._default_max_chunk_words
        self.max_chunk_seconds = self._default_max_chunk_seconds
        self.overlap_seconds = self._default_overlap_seconds
        self.semantic_sensitivity = self._default_semantic_sensitivity
        try:
            profile_name, profile, metrics = estimate_chunk_profile(segments)
        except Exception as exc:
            logger.warning("Adaptive chunk profile failed (%s) — keeping defaults", exc)
            return
        self.max_chunk_words = int(profile["max_chunk_words"])
        self.max_chunk_seconds = float(profile["max_chunk_seconds"])
        self.overlap_seconds = float(profile["overlap_seconds"])
        self.semantic_sensitivity = float(profile["semantic_sensitivity"])
        logger.info(
            "Adaptive chunk profile = %s | max_words=%d max_sec=%.1f overlap=%.1f "
            "sensitivity=%.2f | metrics=%s",
            profile_name, self.max_chunk_words, self.max_chunk_seconds,
            self.overlap_seconds, self.semantic_sensitivity, metrics,
        )

    def chunk(
        self,
        transcription: Any,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ChunkingResult:
        t0 = time.time()

        segments, detected_language, final_video_id = self._extract_segments(
            transcription=transcription, video_id=video_id, language=language,
        )

        # ── ADAPTIVE PROFILE: tune chunk size/overlap to THIS video before packing
        if self.adaptive_profile_enabled:
            self._apply_adaptive_profile(segments)

        if not segments:
            return ChunkingResult(
                video_id=final_video_id, chunks=[], total_segments=0,
                total_chunks=0, language=detected_language,
                processing_seconds=time.time() - t0,
            )

        logger.info(
            "Chunking video_id=%s | segments=%d | language=%s",
            final_video_id, len(segments), detected_language,
        )

        # Pre-split by language
        lang_groups = self._pre_split_on_lang(segments)
        all_chunks: List[SemanticChunk] = []

        for group_lang, group_segments in lang_groups:
            effective_lang = group_lang or detected_language

            sentence_units = self._build_sentence_units(
                segments=group_segments,
                language=effective_lang,
            )

            chunks = self._pack_with_topic_detection(
                sentence_units=sentence_units,
                video_id=final_video_id,
                default_language=effective_lang,
                chunk_offset=len(all_chunks),
            )
            all_chunks.extend(chunks)

        # ── TOPIC-2 : assign topic labels ────────────────────────────
        if self.topic_labeling_enabled:
            self._assign_topic_labels(all_chunks, detected_language)

        # ── SUMMARY-1 : build video summary ─────────────────────────
        video_summary: Optional[VideoSummary] = None
        if self.summarization_enabled and all_chunks:
            duration = 0.0
            if all_chunks:
                last = all_chunks[-1]
                duration = float(
                    getattr(last, "end_time", 0)
                    or (last.get("end_time") if isinstance(last, dict) else 0)
                    or 0
                )
            video_summary = self._video_summarizer.summarize(
                video_id=final_video_id,
                chunks=all_chunks,
                language=detected_language,
                duration_seconds=duration,
            )
            # Inject domain into each chunk's metadata
            if video_summary:
                for c in all_chunks:
                    meta = getattr(c, "metadata", {})
                    if isinstance(meta, dict):
                        meta["domain"] = video_summary.domain
                        meta["sub_domain"] = video_summary.sub_domain

        elapsed = time.time() - t0
        logger.info(
            "Chunking complete: %d chunks in %.2fs | domain=%s",
            len(all_chunks), elapsed,
            video_summary.domain if video_summary else "n/a",
        )

        result = ChunkingResult(
            video_id=final_video_id,
            chunks=all_chunks,
            total_segments=len(segments),
            total_chunks=len(all_chunks),
            language=detected_language,
            processing_seconds=elapsed,
        )

        # Attach video summary to the result (accessed by main.py / metadata.py)
        result.video_summary = video_summary  # type: ignore[attr-defined]

        return result

    # Compatibility aliases
    def create_semantic_chunks(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    def chunk_segments(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    def create_chunks(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    def process_segments(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    def generate_chunks(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    def process(self, *a, **kw) -> ChunkingResult:
        return self.chunk(*a, **kw)

    # ==================================================================
    # Segment extraction & normalization
    # ==================================================================

    def _extract_segments(
        self,
        transcription: Any,
        video_id: Optional[str],
        language: Optional[str],
    ) -> Tuple[List[Any], str, str]:
        raw_segments: List[Any] = []
        detected_language = language or "unknown"

        if transcription is None:
            return [], detected_language, video_id or "unknown_video"

        if isinstance(transcription, TranscriptionResult):
            raw_segments = list(transcription.segments)
            detected_language = language or transcription.language or "unknown"
            final_video_id = video_id or transcription.video_id or "unknown_video"
        elif isinstance(transcription, dict):
            raw_segments = list(
                transcription.get("segments")
                or transcription.get("transcript_segments")
                or transcription.get("items") or []
            )
            detected_language = (
                language
                or transcription.get("language")
                or transcription.get("detected_language") or "unknown"
            )
            final_video_id = (
                video_id or transcription.get("video_id") or "unknown_video"
            )
        elif isinstance(transcription, list):
            raw_segments = list(transcription)
            final_video_id = video_id or "unknown_video"
        else:
            raw_segments = list(
                getattr(transcription, "segments", None)
                or getattr(transcription, "transcript_segments", None)
                or getattr(transcription, "items", None) or []
            )
            detected_language = (
                language
                or getattr(transcription, "language", None)
                or getattr(transcription, "detected_language", None) or "unknown"
            )
            final_video_id = (
                video_id
                or getattr(transcription, "video_id", None) or "unknown_video"
            )

        normalized = []
        for seg in raw_segments:
            parsed = self._to_transcript_segment(
                seg, fallback_language=detected_language
            )
            if parsed is not None:
                normalized.append(parsed)

        if detected_language == "unknown" and normalized:
            detected_language = self.detector.dominant_language(
                [s.text for s in normalized]
            ).code

        return normalized, detected_language, final_video_id

    def _to_transcript_segment(
        self,
        seg: Any,
        fallback_language: str = "unknown",
    ) -> Optional[Any]:
        if isinstance(seg, TranscriptSegment):
            if not seg.text.strip():
                return None
            if (seg.language or "unknown") == "unknown":
                seg = TranscriptSegment(
                    text=seg.text,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    language=self.detector.detect_language(
                        seg.text, hint=fallback_language
                    ),
                    speaker=seg.speaker,
                    confidence=seg.confidence,
                    words=seg.words,
                )
            return seg

        if isinstance(seg, dict):
            text = _safe_str(
                seg.get("text") or seg.get("content") or seg.get("transcript")
            )
            start_time = _safe_float(seg.get("start_time", seg.get("start", 0.0)))
            end_time = _safe_float(seg.get("end_time", seg.get("end", 0.0)))
            language = _safe_str(seg.get("language"), fallback_language) or fallback_language
            speaker = _safe_str(seg.get("speaker"), "SPEAKER_00")
            confidence = _safe_float(seg.get("confidence", 1.0), 1.0)
            words = seg.get("words") or []
        else:
            text = _safe_str(
                getattr(seg, "text", None)
                or getattr(seg, "content", None)
                or getattr(seg, "transcript", None)
            )
            start_time = _safe_float(
                getattr(seg, "start_time", getattr(seg, "start", 0.0))
            )
            end_time = _safe_float(
                getattr(seg, "end_time", getattr(seg, "end", 0.0))
            )
            language = (
                _safe_str(getattr(seg, "language", None), fallback_language)
                or fallback_language
            )
            speaker = _safe_str(getattr(seg, "speaker", None), "SPEAKER_00")
            confidence = _safe_float(getattr(seg, "confidence", 1.0), 1.0)
            words = getattr(seg, "words", []) or []

        text = text.strip()
        if not text:
            return None
        if end_time < start_time:
            end_time = start_time
        if language in ("unknown", ""):
            language = self.detector.detect_language(text, hint=fallback_language)

        return TranscriptSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            language=language,
            speaker=speaker,
            confidence=confidence,
            words=words,
        )

    def _serialize_word(self, word: Any) -> Optional[Dict[str, Any]]:
        if isinstance(word, dict):
            token = _safe_str(word.get("word") or word.get("text") or word.get("token"))
            start = word.get("start", word.get("start_time"))
            end = word.get("end", word.get("end_time"))
            score = word.get("score", word.get("confidence", word.get("probability")))
        else:
            token = _safe_str(
                getattr(word, "word", None)
                or getattr(word, "text", None)
                or getattr(word, "token", None)
            )
            start = getattr(word, "start", getattr(word, "start_time", None))
            end = getattr(word, "end", getattr(word, "end_time", None))
            score = (
                getattr(word, "score", None)
                or getattr(word, "confidence", None)
                or getattr(word, "probability", None)
            )

        token = token.strip()
        if not token:
            return None

        item: Dict[str, Any] = {"word": token}
        if start is not None:
            item["start"] = _safe_float(start)
        if end is not None:
            item["end"] = _safe_float(end)
        if score is not None:
            try:
                item["score"] = float(score)
            except (TypeError, ValueError):
                pass
        return item

    def _serialize_timeline_segment(self, seg: Any) -> Dict[str, Any]:
        words_raw = _seg_get(seg, "words", []) or []
        words: List[Dict[str, Any]] = []
        for word in words_raw[:128]:
            parsed = self._serialize_word(word)
            if parsed is not None:
                words.append(parsed)

        start_time = _safe_float(_seg_get(seg, "start_time"))
        end_time = _safe_float(_seg_get(seg, "end_time"))
        return {
            "text": _seg_get(seg, "text", "").strip(),
            "start_time": start_time,
            "end_time": end_time,
            "start_ms": int(start_time * 1000),
            "end_ms": int(end_time * 1000),
            "language": _seg_get(seg, "language") or "unknown",
            "speaker": _seg_get(seg, "speaker") or "SPEAKER_00",
            "confidence": _safe_float(_seg_get(seg, "confidence", 1.0), 1.0),
            "words": words,
        }

    # ==================================================================
    # Pre-split by language (BUG-2 fix from v3 preserved)
    # ==================================================================

    def _pre_split_on_lang(
        self,
        segments: List[Any],
    ) -> List[Tuple[str, List[Any]]]:
        if not segments:
            return []

        groups: List[Tuple[str, List[Any]]] = []
        current_lang = _seg_get(segments[0], "language") or "unknown"
        current_group: List[Any] = [segments[0]]

        for seg in segments[1:]:
            seg_lang = _seg_get(seg, "language") or "unknown"
            if seg_lang != current_lang and seg_lang != "unknown" and current_lang != "unknown":
                groups.append((current_lang, current_group))
                current_lang = seg_lang
                current_group = [seg]
            else:
                current_group.append(seg)

        if current_group:
            groups.append((current_lang, current_group))

        # Merge short groups (< 2 segs) into adjacent if gap is small
        merged: List[Tuple[str, List[Any]]] = []
        for i, (lang, grp) in enumerate(groups):
            if len(grp) < 2 and merged:
                prev_lang, prev_grp = merged[-1]
                last_end = _safe_float(_seg_get(prev_grp[-1], "end_time"))
                first_start = _safe_float(_seg_get(grp[0], "start_time"))
                gap = max(0.0, first_start - last_end)
                if gap <= MAX_MERGE_GAP_S:
                    merged[-1] = (prev_lang, prev_grp + grp)
                    continue
            merged.append((lang, grp))

        return merged

    # ==================================================================
    # Build sentence units (complete sentences, language-aware)
    # ==================================================================

    def _build_sentence_units(
        self,
        segments: List[Any],
        language: str,
    ) -> List[List[Any]]:
        """
        Group consecutive segments into complete sentence units.
        Never splits in the middle of a sentence.
        Detects topic-shift markers as hard boundaries.
        """
        if not segments:
            return []
        if not self.merge_short_segments:
            return [[s] for s in segments]

        gap_limit = _gap_threshold(language, fallback=self.max_gap_seconds)
        units: List[List[Any]] = []
        current: List[Any] = []

        for seg in segments:
            if not current:
                current = [seg]
                continue

            previous = current[-1]
            gap = max(
                0.0,
                _safe_float(_seg_get(seg, "start_time"))
                - _safe_float(_seg_get(previous, "end_time")),
            )
            current_words = sum(
                _word_count(_seg_get(s, "text", "")) for s in current
            )
            seg_text = _seg_get(seg, "text", "").strip()

            should_flush = False
            flush_reason = ""

            # 1. Speaker change
            if self.split_on_speaker_change and _speaker_changed_meaningfully(
                previous, seg
            ):
                should_flush = True
                flush_reason = "speaker_change"

            # 2. Language switch
            elif _language_switched(previous, seg):
                should_flush = True
                flush_reason = "language_switch"

            # 3. Topic shift marker (new in v4.0)
            elif self._topic_shift_detector.is_topic_shift(seg_text, language):
                # Only flush if current unit has enough content
                if current_words >= self.min_chunk_words // 2:
                    should_flush = True
                    flush_reason = "topic_shift_marker"

            # 4. Long pause (gap threshold)
            elif gap > gap_limit:
                should_flush = True
                flush_reason = "gap"

            # 5. Sentence boundary (only if we have enough words)
            elif (
                _is_sentence_end(
                    _seg_get(previous, "text", ""),
                    _seg_get(previous, "language") or language,
                )
                and current_words >= self.min_chunk_words // 2
            ):
                should_flush = True
                flush_reason = "sentence_end"

            if should_flush:
                units.append(current)
                current = [seg]
                logger.debug(
                    "Sentence unit flushed: %d segs, reason=%s",
                    len(units[-1]),
                    flush_reason,
                )
            else:
                current.append(seg)

        if current:
            units.append(current)

        return units

    # ==================================================================
    # Pack units into chunks with topic detection (v4.0 core)
    # ==================================================================

    def _pack_with_topic_detection(
        self,
        sentence_units: List[List[Any]],
        video_id: str,
        default_language: str,
        chunk_offset: int = 0,
    ) -> List[SemanticChunk]:
        """
        Groups sentence units into chunks.

        Boundary signals (in order):
            (a) TopicShiftDetector marker at start of unit
            (b) SemanticSplitDetector distance spike
            (c) Hard limits (words / seconds)

        Ensures complete sentences are never split.
        """
        if not sentence_units:
            return []

        # ── Semantic split points (MiniLM) ───────────────────────────
        semantic_splits: Set[int] = set()
        if self._semantic_detector and len(sentence_units) > 2:
            unit_texts = [
                " ".join(_seg_get(s, "text", "") for s in unit).strip()
                for unit in sentence_units
            ]
            splits = self._semantic_detector.find_split_points(
                unit_texts, sensitivity=self.semantic_sensitivity
            )
            semantic_splits = set(splits)
            if semantic_splits:
                logger.debug(
                    "Semantic split points at unit indices: %s", semantic_splits
                )

        # ── Pack into groups ─────────────────────────────────────────
        chunk_groups: List[Tuple[List[List[Any]], str]] = []
        buffer_units: List[List[Any]] = []

        for idx, unit in enumerate(sentence_units):
            unit_text = " ".join(
                _seg_get(s, "text", "") for s in unit
            ).strip()

            # Is this unit a topic-shift boundary?
            unit_lang = (
                _seg_get(unit[0], "language") or default_language
                if unit else default_language
            )
            is_topic_shift = (
                self._topic_shift_detector.is_topic_shift(unit_text, unit_lang)
                and bool(buffer_units)
            )

            # Is this a semantic boundary?
            is_semantic_boundary = (idx in semantic_splits) and bool(buffer_units)

            # Would adding this unit exceed size limits?
            candidate = buffer_units + [unit]
            exceeds = self._would_exceed_limits(candidate)

            if is_topic_shift or is_semantic_boundary or exceeds:
                if buffer_units:
                    if is_topic_shift:
                        method = "topic_shift"
                    elif is_semantic_boundary and not exceeds:
                        method = "semantic"
                    else:
                        method = "rule_based"

                    chunk_groups.append((buffer_units, method))
                    overlap = self._make_overlap_units(buffer_units)
                    buffer_units = overlap + [unit] if overlap else [unit]
                else:
                    # Unit alone exceeds limits → hard limit
                    chunk_groups.append(([unit], "hard_limit"))
                    buffer_units = []
            else:
                buffer_units.append(unit)

        if buffer_units:
            chunk_groups.append((buffer_units, "rule_based"))

        # ── Build SemanticChunk objects ──────────────────────────────
        chunks: List[SemanticChunk] = []
        for units, split_method in chunk_groups:
            chunk = self._build_chunk(
                units=units,
                chunk_index=chunk_offset + len(chunks),
                video_id=video_id,
                default_language=default_language,
                split_method=split_method,
            )
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    # ==================================================================
    # Limit check
    # ==================================================================

    def _would_exceed_limits(self, units: List[List[Any]]) -> bool:
        segments = [seg for unit in units for seg in unit]
        if not segments:
            return False
        total_words = sum(
            _word_count(_seg_get(seg, "text", "")) for seg in segments
        )
        first_start = _safe_float(_seg_get(segments[0], "start_time"))
        last_end = _safe_float(_seg_get(segments[-1], "end_time"))
        duration = last_end - first_start
        return total_words > self.max_chunk_words or duration > self.max_chunk_seconds

    # ==================================================================
    # Overlap (BUG-1 fix from v3 preserved)
    # ==================================================================

    def _make_overlap_units(
        self,
        units: List[List[Any]],
    ) -> List[List[Any]]:
        if self.overlap_seconds <= 0 or not units:
            return []

        last_unit_speaker = (
            _seg_get(units[-1][-1], "speaker") if units[-1] else None
        )
        selected: List[List[Any]] = []
        accumulated = 0.0

        for unit in reversed(units):
            if not unit:
                continue
            unit_speaker = _seg_get(unit[-1], "speaker")
            if selected and unit_speaker != last_unit_speaker:
                break
            first_start = _safe_float(_seg_get(unit[0], "start_time"))
            last_end = _safe_float(_seg_get(unit[-1], "end_time"))
            raw_duration = last_end - first_start
            unit_duration = max(raw_duration, 0.1)  # BUG-1 fix

            if accumulated + unit_duration > self.overlap_seconds and selected:
                break
            selected.insert(0, unit)
            accumulated += unit_duration
            if accumulated >= self.overlap_seconds:
                break

        return selected

    # ==================================================================
    # Build one SemanticChunk from a list of units
    # ==================================================================

    def _build_chunk(
        self,
        units: List[List[Any]],
        chunk_index: int,
        video_id: str,
        default_language: str,
        split_method: str = "rule_based",
    ) -> Optional[SemanticChunk]:
        segments = [seg for unit in units for seg in unit]
        if not segments:
            return None

        text_parts = [
            _seg_get(seg, "text", "").strip()
            for seg in segments
            if _seg_get(seg, "text", "").strip()
        ]
        combined_text = " ".join(text_parts).strip()
        if not combined_text:
            return None

        start_time = _safe_float(_seg_get(segments[0], "start_time"))
        end_time = _safe_float(_seg_get(segments[-1], "end_time"))
        duration = max(0.0, end_time - start_time)

        # Reject meaningless tiny chunks
        if duration < 0.5 and _word_count(combined_text) < 3:
            return None

        # Language distribution
        lang_dist, is_multilingual = self._compute_chunk_language_distribution(
            segments, default_language,
        )
        chunk_language = (
            max(lang_dist, key=lang_dist.get) if lang_dist else default_language
        )

        # Speaker votes
        speaker_votes: Dict[str, float] = {}
        for seg in segments:
            seg_start = _safe_float(_seg_get(seg, "start_time"))
            seg_end = _safe_float(_seg_get(seg, "end_time"))
            dur = max(0.0, seg_end - seg_start)
            spk = _seg_get(seg, "speaker") or "SPEAKER_00"
            speaker_votes[spk] = speaker_votes.get(spk, 0.0) + max(dur, 0.1)
        dominant_speaker = (
            max(speaker_votes, key=speaker_votes.get)
            if speaker_votes else "SPEAKER_00"
        )

        # Normalize text
        normalized = self.normalizer.process(combined_text, language=chunk_language)
        norm_text = getattr(normalized, "normalized_text", combined_text)
        search_text = getattr(normalized, "search_text", combined_text)[:1024]

        # Filter tags
        filter_tags: List[str] = []
        lang_code = chunk_language[:2].lower() if chunk_language else "en"
        if lang_code == "ar":
            filter_tags.append("arabic")
        elif lang_code == "fr":
            filter_tags.append("french")
        else:
            filter_tags.append("english")
        if is_multilingual:
            filter_tags.append("bilingual")
        if len(speaker_votes) > 1:
            filter_tags.append("multi_speaker")
        else:
            filter_tags.append("single_speaker")
        filter_tags.append(f"split:{split_method}")

        # Language spans
        language_spans: List[Dict[str, Any]] = []
        current_lang: Optional[str] = None
        current_span_start: Optional[float] = None
        current_span_texts: List[str] = []

        for seg in segments:
            seg_lang = _seg_get(seg, "language") or chunk_language
            seg_start = _safe_float(_seg_get(seg, "start_time"))
            seg_text = _seg_get(seg, "text", "").strip()

            if seg_lang != current_lang:
                if current_lang is not None and current_span_start is not None:
                    language_spans.append({
                        "language": current_lang,
                        "start_ms": int(current_span_start * 1000),
                        "end_ms": int(seg_start * 1000),
                        "text": " ".join(current_span_texts),
                    })
                current_lang = seg_lang
                current_span_start = seg_start
                current_span_texts = [seg_text]
            else:
                current_span_texts.append(seg_text)

        if current_lang is not None and current_span_start is not None:
            language_spans.append({
                "language": current_lang,
                "start_ms": int(current_span_start * 1000),
                "end_ms": int(end_time * 1000),
                "text": " ".join(current_span_texts),
            })

        # Segmentation signals
        start_reason_map = {
            "topic_shift":  "topic_change",
            "semantic":     "semantic_shift",
            "rule_based":   "pause",
            "hard_limit":   "max_duration",
        }
        segmentation_signals: Dict[str, Any] = {
            "start_reason": start_reason_map.get(split_method, "unknown"),
            "end_reason": "pause",
            "split_method": split_method,
            "token_count": _word_count(combined_text),
            "force_cut": split_method == "hard_limit",
            "boundary_score": 0.0,
        }

        cid = _chunk_id(video_id, start_time, end_time, combined_text)

        metadata: Dict[str, Any] = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "word_count": _word_count(combined_text),
            "chunk_type": "speech",
            "topic": "general",      # will be updated by _assign_topic_labels
            "search_text": search_text,
            "normalized_text": norm_text,
            "dominant_speaker": dominant_speaker,
            "segment_count": len(segments),
            "language_distribution": lang_dist,
            "is_multilingual": is_multilingual,
            "split_method": split_method,
            "filter_tags": filter_tags,
            "language_spans": language_spans,
            "timeline_segments": [
                self._serialize_timeline_segment(seg)
                for seg in segments
            ],
            "segmentation_signals": segmentation_signals,
            # v4.0 additions
            "domain": "",          # filled after summarization
            "sub_domain": "",      # filled after summarization
        }

        return SemanticChunk(
            chunk_id=cid,
            video_id=video_id,
            text=combined_text,
            original_text=combined_text,
            normalized_text=norm_text,
            search_text=search_text,
            start_time=start_time,
            end_time=end_time,
            language=chunk_language,
            speaker=dominant_speaker,
            topic="general",        # updated by _assign_topic_labels
            chunk_type="speech",
            chunk_index=chunk_index,
            metadata=metadata,
        )

    # ==================================================================
    # TOPIC-2 : Assign topic labels to all chunks
    # ==================================================================

    def _assign_topic_labels(
        self,
        chunks: List[SemanticChunk],
        language: str,
    ) -> None:
        """
        Assign a topic label to each chunk in-place.
        Batches topic-shift boundaries to minimize LLM calls.
        """
        for chunk in chunks:
            text = (
                getattr(chunk, "text", "")
                or (chunk.get("text") if isinstance(chunk, dict) else "")
                or ""
            ).strip()

            lang = (
                getattr(chunk, "language", language)
                or (chunk.get("language") if isinstance(chunk, dict) else language)
                or language
            )

            if not text:
                continue

            topic = self._topic_labeler.label(text, lang)

            # Write to the chunk object
            if hasattr(chunk, "topic"):
                chunk.topic = topic
            elif isinstance(chunk, dict):
                chunk["topic"] = topic

            # Also write to metadata
            meta = getattr(chunk, "metadata", {})
            if isinstance(meta, dict):
                meta["topic"] = topic

    # ==================================================================
    # Language distribution (BUG-3 fix from v3 preserved)
    # ==================================================================

    def _compute_chunk_language_distribution(
        self,
        segments: List[Any],
        default_language: str,
    ) -> Tuple[Dict[str, float], bool]:
        duration_per_lang: Dict[str, float] = {}
        total_duration = 0.0

        for seg in segments:
            lang = _seg_get(seg, "language") or default_language
            if lang == "unknown":
                lang = default_language

            seg_start = _safe_float(_seg_get(seg, "start_time"))
            seg_end = _safe_float(_seg_get(seg, "end_time"))
            dur = max(0.0, seg_end - seg_start)
            duration_per_lang[lang] = duration_per_lang.get(lang, 0.0) + dur
            total_duration += dur

        if total_duration <= 0:
            return {default_language: 1.0}, False

        distribution = {
            lang: round(dur / total_duration, 4)
            for lang, dur in duration_per_lang.items()
        }
        significant_langs = [
            lang for lang, share in distribution.items()
            if share > MULTILINGUAL_THRESHOLD
        ]
        is_multilingual = len(significant_langs) > 1

        return distribution, is_multilingual


# ══════════════════════════════════════════════════════════════════════
# Module-level singleton
# ══════════════════════════════════════════════════════════════════════

_chunker_instance: Optional[SemanticChunker] = None


def get_chunker(**kwargs: Any) -> SemanticChunker:
    """Return (and lazily create) a module-level SemanticChunker singleton."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = SemanticChunker(**kwargs)
    return _chunker_instance


__all__ = [
    "SemanticChunker",
    "VideoSummary",
    "VideoSummarizer",
    "TopicShiftDetector",
    "TopicLabeler",
    "DOMAIN_TAXONOMY",
    "detect_domain_heuristic",
    "get_chunker",
]