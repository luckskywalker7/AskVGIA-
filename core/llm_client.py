# core/llm_client.py  — v7.0 (MULTI-TIMESTAMP + CONVERSATION HISTORY + CONFIDENCE SCORING)
# ═══════════════════════════════════════════════════════════════════════
#
#  v7.0 NEW FIXES:
#
#    MULTI-TS-1     _extract_timestamp_seconds() replaced by
#                   _extract_all_timestamps() which returns a LIST of
#                   all timestamps found in the question (not just the
#                   first). Enables answers to multi-part questions like
#                   "what happened at 2:00 and 5:30?".
#
#    MULTI-TS-2     synthesize_answer() and stream_answer() now merge
#                   results from ALL requested timestamps, deduplicated,
#                   then sorted chronologically before sending to the LLM.
#
#    MULTI-TS-3     _build_user_prompt() coverage_note lists ALL
#                   requested timestamps so the LLM knows exactly which
#                   moments to address.
#
#    CONV-1         ConversationTurn dataclass — holds one Q&A pair with
#                   optional intent + language metadata.
#
#    CONV-2         synthesize_answer() and stream_answer() accept an
#                   optional conversation_history: List[ConversationTurn].
#                   The last N turns (default 3) are injected into the
#                   Ollama messages list as real assistant/user turns so
#                   the LLM can resolve pronouns and follow-up references.
#
#    CONV-3         _build_history_messages() helper — formats history
#                   into Ollama message dicts, capped at max_turns to
#                   stay within the context window.
#
#    CONF-1         _confidence_score() — computes a 0-1 score from
#                   the top citation's relevance and how many citations
#                   were found. Low score → uncertain answer.
#
#    CONF-2         _postprocess_answer() now appends a natural-language
#                   uncertainty disclaimer when confidence < LOW_CONF_THRESHOLD
#                   (0.35). The LLM answer is kept but the user is told
#                   it may not be accurate.
#
#    CONF-3         _extractive_fallback() also carries a confidence score
#                   so the UI can decide whether to show a "low confidence"
#                   badge even on fallback answers.
#
#  v6.0 improvements kept (VIDEO-AWARE-1..5, STREAM-1)
#  Previous fixes kept: BUG-5, FIX-1..4, and all 9 fixes from v6.1
#
# ═══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

try:
    from config.settings import settings
except Exception:
    settings = None

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300
DEFAULT_NUM_PREDICT = 384          # v6.0: 384 → 512 → back to 384: tighter budget = faster generation
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.85
DEFAULT_REPEAT_PENALTY = 1.05  # lower = more natural sentence flow

PREFERRED_MODEL_PRIORITY = [
     "qwen3.5:2b", "qwen3:4b", "qwen3:8b", "qwen3:1.7b",
    "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b",
    "mistral:7b", "llama3.1:8b", "llama3.2:3b", "phi3:3.8b", "gemma2:2b",
]

PREFERRED_FAMILIES = [
    "qwen3.5", "qwen3", "qwen2.5", "qwen",
    "mistral", "llama3", "phi", "gemma",
]

# ── Arabic ordinal → seconds map (used by timestamp detection) ──────
_AR_ORDINAL_TO_SEC: Dict[str, float] = {
    "الأولى": 60, "الأول": 60,
    "الثانية": 120, "الثاني": 120,
    "الثالثة": 180, "الثالث": 180,
    "الرابعة": 240, "الرابع": 240,
    "الخامسة": 300, "الخامس": 300,
    "السادسة": 360, "السادس": 360,
    "السابعة": 420, "السابع": 420,
    "الثامنة": 480, "الثامن": 480,
    "التاسعة": 540, "التاسع": 540,
    "العاشرة": 600, "العاشر": 600,
}
# Regex for "في الدقيقة الثانية" / "عند الدقيقة 3" / "بعد الدقيقة الخامسة"
_AR_ORDINAL_TS_RE = re.compile(
    r"(دقيقة|دقائق)\s*"
    r"(الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة"
    r"|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر"
    r"|\d+)",
    re.IGNORECASE,
)

def _ar_ordinal_to_seconds(question: str) -> Optional[float]:
    """Convert Arabic ordinal minute reference to seconds, or None."""
    m = _AR_ORDINAL_TS_RE.search(question)
    if not m:
        return None
    val = m.group(2).strip()
    if val.isdigit():
        return float(val) * 60
    return _AR_ORDINAL_TO_SEC.get(val)


# ─────────────────────────────────────────────────────────────────────
# INTENT PATTERNS  (ordered from most-specific to least-specific)
# Each pattern is tested in _detect_intent; first match wins.
# ─────────────────────────────────────────────────────────────────────

# TRANSLATE — always highest priority after timestamp
TRANSLATE_PATTERNS = [
    r"\btranslat\b", r"\btraduction\b", r"\bترجم\b", r"\bترجمة\b",
    r"\bin arabic\b", r"\bin french\b", r"\bin english\b",
    r"\ben arabe\b", r"\ben français\b", r"\ben anglais\b",
    r"\bبالعربية\b", r"\bبالفرنسية\b", r"\bبالإنجليزية\b",
]

# SPEAKER — who/how-many people speak; must beat LIST (كم) and FACTUAL (من)
SPEAKER_PATTERNS = [
    # English — "how many speakers/people/voices [are there/in this/...]"
    r"\bhow many (speaker\w*|person\w*|people|voice\w*|character\w*)\b",
    r"\bnumber of (speaker\w*|person\w*|people|voice\w*)\b",
    r"\bcount.*speaker\w*\b",
    r"\bwho (is|are) (speak\w*|talking|saying|the speaker|the voice)\b",
    r"\bwho (speak|talk|narrat|present)\w*\b",
    r"\bwho is speaking\b",
    r"\bhow many people (speak|talk|are speaking)\b",
    # French
    r"\bcombien.*interlocuteur\b", r"\bcombien.*personne\b",
    r"\bqui parle\b", r"\bqui est.*voix\b",
    # Arabic — count
    r"كم\s*(من\s*)?(متحدث|متحدثون|متحدثين|شخص|أشخاص|صوت|أصوات)",
    r"عدد\s*(المتحدثين|الأشخاص|الأصوات|المتكلمين)",
    r"كم\s*(شخص|متحدث|صوت)\b",
    # Arabic — who is speaking
    r"من\s+(يتحدث|يتكلم|يتحدثون|المتحدث|المتحدثون)",
    r"من\s+هم\s+(الأشخاص|المتحدثون|المتكلمون)",
    r"(المتحدث|المتحدثون|الأشخاص)\s+في\s+الفيديو",
]

# VISUAL — answered from frames, not speech
VISUAL_PATTERNS = [
    r"\bwear\b", r"\bwearing\b", r"\bwore\b", r"\boutfit\b",
    r"\bdress\b", r"\bdressed\b", r"\bclothes\b", r"\bclothing\b",
    r"\bshirt\b", r"\bjacket\b", r"\bsuit\b", r"\bcostume\b",
    r"\bappear\b", r"\bappearance\b", r"\blooks? like\b",
    r"\bhair\b", r"\bglasses\b", r"\bbeard\b",
    r"\bcolor\b", r"\bcolou?r\b",
    r"\bred\b.*\b(dress|shirt|outfit|jacket|clothes)\b",
    r"\b(dress|shirt|outfit|jacket|clothes)\b.*\bred\b",
    # Arabic visual
    r"\bملابس\b", r"\bيلبس\b", r"\bيرتدي\b", r"\bمظهر\b",
    r"\bشكل\b.*\b(الشخص|المتحدث)\b",
]

# YES/NO — questions that start with yes/no auxiliaries
YES_NO_PATTERNS = [
    r"^did\b", r"^does\b", r"^do\b", r"^is\b", r"^are\b",
    r"^was\b", r"^were\b", r"^can\b", r"^could\b", r"^will\b",
    r"^has\b", r"^have\b",
    r"^est-ce que\b", r"^est-il\b",
    r"^هل\b",
]

# SUMMARY — what is the video about / overview
SUMMARY_PATTERNS = [
    # English
    r"\bsummar",           # matches: summary, summarize, summarise, summarization
    r"\boverall\b", r"\bmain (point|topic|idea|theme)\b",
    r"\btopic\b", r"\btheme\b", r"\bexplain\b", r"\bdescribe\b",
    r"\bwhat (is|does|did).{0,25}(video|talk|cover|about)\b",
    r"\btell me about\b", r"\bgive.*overview\b",
    # French
    r"\bde quoi\b", r"\bsujet\b", r"\brésume\b", r"\bparle\b",
    r"\bde quoi.*vidéo\b",
    # Arabic — what is the video about
    r"(عن\s+ماذا|ماذا\s+يتناول|بماذا\s+يتحدث|ما\s+الذي\s+يدور|يدور\s+حوله)",
    r"ما\s+(موضوع|محتوى|مضمون)\s*(ال)?فيديو",
    r"(اشرح|صف|لخّص|لخص)\s*(ما|محتوى|الفيديو)",
    r"ما\s+هو\s+الفيديو\b",
    r"\bيتحدث\s+عن\b",
    r"\bما\s+موضوع\b",
    r"\bما\s+هو\b",
]

# TEMPORAL — when did something happen
TEMPORAL_PATTERNS = [
    r"\bwhen\b", r"\bwhat time\b", r"\bhow long\b",
    r"\bbefore\b", r"\bafter\b",
    r"\bfirst\b", r"\blast\b", r"\bduring\b",
    r"\bminute\b", r"\btimestamp\b", r"\bhour\b",
    r"\bmoment\b", r"\bbeginning\b", r"\bsequence\b",
    r"\bquand\b", r"\bà quel moment\b", r"\bdébut\b", r"\bfin\b",
    r"\bavant\b", r"\baprès\b", r"\bpremier\b", r"\bdernier\b",
    r"\bpendant\b", r"\bcombien de temps\b", r"\bdurant\b",
    r"\bمتى\b", r"\bقبل\b", r"\bبعد\b",
    r"\bأول\b", r"\bآخر\b", r"\bبداية\b", r"\bنهاية\b",
    r"\bخلال\b", r"\bمدة\b", r"\bوقت\b",
    # Arabic ordinal minute references trigger temporal (will be refined to timestamp_lookup)
    r"(دقيقة|دقائق)\s*(الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+)",
]

# LIST — enumerate / how many (non-speaker)
LIST_PATTERNS = [
    r"\blist\b", r"\ball\b", r"\bwhich\b", r"\bwhat are\b",
    r"\benumerate\b", r"\bgive me (all|a list)\b",
    r"\bhow many\b", r"\bevery\b", r"\beach\b",
    r"\bliste\b", r"\btous\b", r"\btoutes\b", r"\bquels\b",
    r"\bquelles\b", r"\bénumère\b", r"\bcombien\b", r"\bchaque\b",
    r"\bاذكر\b", r"\bما هي\b", r"\bجميع\b",
    r"\bكم\b", r"\bعدد\b",
]

# FACTUAL — catch-all for specific question words
FACTUAL_PATTERNS = [
    r"\bdid\b", r"\bdoes\b", r"\bis\b", r"\bwas\b", r"\bwere\b",
    r"\bwhat\b", r"\bwho\b", r"\bwhy\b", r"\bhow\b",
    r"\best-ce que\b", r"\bqui\b", r"\bquoi\b", r"\bpourquoi\b",
    r"\bcomment\b", r"\bهل\b", r"\bما\b", r"\bلماذا\b", r"\bكيف\b",
    r"\bمن\b", r"\bماذا\b", r"\bأين\b",
]


# ─────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LLMAnswer:
    answer: str
    model: Optional[str]
    used_fallback: bool
    citations: List[Dict[str, Any]]
    intent: str
    language_hint: str
    raw: Optional[Dict[str, Any]] = None
    confidence: float = 1.0          # CONF-1: 0.0–1.0 answer confidence score


# CONV-1: Represents one prior Q&A turn for conversation history
@dataclass
class ConversationTurn:
    question: str
    answer: str
    intent: str = "factual"
    language_hint: str = "en"


# CONF-1: Answers below this threshold get an uncertainty disclaimer appended
LOW_CONF_THRESHOLD = 0.22   # was 0.35 — lower to avoid false positives on paraphrased questions


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────


def _clean_visual_text_for_llm(raw):
    """Strip raw JSON/dict/metadata leakage from visual frame text before feeding to LLM."""
    import re as _re2
    if not raw:
        return raw
    s = raw.strip()
    has_text_key = ("'text'" in s) or '"text"' in s
    if s.startswith(("{", "[")) or has_text_key:
        m = _re2.search(r"""['"](text)['"]\s*:\s*['"]([^'\"]{4,})['"]""", s)
        if m:
            return m.group(2).strip()
        m = _re2.search(r"""\btext\b\s*:\s*(.+?)(?:\s+\w+\s*[:{]|$)""", s, _re2.IGNORECASE)
        if m:
            cand = m.group(1).strip().strip("'\"")
            if len(cand) > 4:
                return cand
        return ""
    s = _re2.sub(r"\s*\{[^}]{0,200}\}\s*$", "", s).strip()
    s = _re2.sub(r"\s+(authority|confidence|language|score|confidence_level)\s*:\s*\S+.*$",
                 "", s, flags=_re2.IGNORECASE).strip()
    return s

def _cfg(name: str, default: Any) -> Any:
    if settings is None:
        return default
    return getattr(settings, name, default)


def _get_ollama_base_url() -> str:
    host = _cfg("ollama_host", None)
    port = _cfg("ollama_port", None)
    url = _cfg("ollama_url", None) or _cfg("ollama_base_url", None)
    if isinstance(url, str) and url.strip():
        return url.rstrip("/")
    if host and port:
        return f"http://{host}:{port}"
    return DEFAULT_OLLAMA_URL


def _timeout() -> int:
    return int(_cfg("ollama_timeout", DEFAULT_TIMEOUT))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _format_ts(seconds: Any) -> str:
    try:
        s = max(0.0, float(seconds))
    except Exception:
        return "00:00"
    total = int(round(s))
    h, remainder = divmod(total, 3600)
    m, sec = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _score_of(result: Any) -> float:
    for key in ("rerank_score", "score", "hybrid_score", "dense_score", "sparse_score"):
        val = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
        if val is not None:
            return _safe_float(val, 0.0)
    return 0.0


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_visual_field(obj: Any, name: str, default: Any = None) -> Any:
    """
    STEP-0 FIX: read a visual field from the top level, falling back to the
    nested ``metadata`` dict. The vision pipeline stores spatial_layout /
    object_counts / board_text / screen_text inside metadata, and
    _normalize_result only hoists caption-like fields, so a plain _get() would
    return nothing for the structured fields.
    """
    val = _get(obj, name, None)
    if val not in (None, "", [], {}):
        return val
    meta = _get(obj, "metadata", None)
    if isinstance(meta, str):
        try:
            import json as _json
            meta = _json.loads(meta)
        except Exception:
            meta = None
    if isinstance(meta, dict):
        mval = meta.get(name)
        if mval not in (None, "", [], {}):
            return mval
    return default


def _result_text(result: Any) -> str:
    return (
        _normalize_text(_get(result, "text"))
        or _normalize_text(_get(result, "original_text"))
        or _normalize_text(_get(result, "search_text"))
    )


def _collect_key_points(result: Any) -> List[str]:
    kp = _get(result, "key_points")
    if kp is None:
        return []
    if isinstance(kp, list):
        return [str(x).strip() for x in kp if str(x).strip()]
    if isinstance(kp, str):
        lines = [x.strip("-• \t\r\n") for x in kp.splitlines()]
        return [x for x in lines if x]
    return []


_MATCH_STOPWORDS = {
    "when", "what", "time", "did", "does", "do", "is", "was", "were", "the", "a", "an",
    "mention", "mentioned", "mentions", "say", "says", "said", "talk", "talked", "about",
    "this", "that", "these", "those", "there", "their", "video", "speaker", "moment",
    "quand", "quel", "quelle", "quelles", "quels", "moment", "temps", "mentionne", "parle",
    "dit", "dans", "cette", "vidéo", "video", "est", "sont", "les", "des", "une", "un",
    "متى", "ما", "ماذا", "قال", "ذكر", "تحدث", "هذا", "هذه", "ذلك", "هناك", "فيديو",
}


def _normalize_match_token(token: Any) -> str:
    token = re.sub(r"[^\w\u0600-\u06FF]+", "", str(token or "").strip().lower())
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


def _tokenize_for_match(text: Any) -> List[str]:
    raw = re.findall(r"[\w\u0600-\u06FF']+", str(text or "").lower())
    out: List[str] = []
    for tok in raw:
        norm = _normalize_match_token(tok)
        if len(norm) >= 3:
            out.append(norm)
    return out


def _extract_question_keywords(question: str) -> List[str]:
    keywords: List[str] = []
    seen = set()
    for tok in _tokenize_for_match(question):
        if tok in _MATCH_STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords[:12]


def _extract_timeline_segments(result: Any) -> List[Dict[str, Any]]:
    meta = _get(result, "metadata", {}) or {}
    timeline = meta.get("timeline_segments") or meta.get("segments") or []
    return [seg for seg in timeline if isinstance(seg, dict)]


def _score_timeline_segment(seg: Dict[str, Any], keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text = str(seg.get("text") or "")
    if not text.strip():
        return 0.0
    normalized_text = " ".join(_tokenize_for_match(text))
    token_set = set(normalized_text.split())
    score = 0.0
    for kw in keywords:
        if kw in token_set:
            score += 4.0
        elif kw in normalized_text:
            score += 2.5
    if keywords and len(token_set.intersection(keywords)) == len(keywords):
        score += 1.5
    return score


def _refine_with_word_timings(words: List[Dict[str, Any]], keywords: List[str]) -> Optional[Dict[str, Any]]:
    if not words or not keywords:
        return None
    for word in words:
        token = _normalize_match_token(word.get("word") or word.get("text") or word.get("token"))
        if token and token in keywords:
            start = word.get("start", word.get("start_time"))
            end = word.get("end", word.get("end_time", start))
            if start is not None:
                start_f = _safe_float(start)
                end_f = _safe_float(end, start_f)
                return {
                    "start_time": start_f,
                    "end_time": max(start_f, end_f),
                    "matched_word": token,
                }
    return None


def _find_precise_span(result: Any, question: str) -> Optional[Dict[str, Any]]:
    keywords = _extract_question_keywords(question)
    if not keywords:
        return None
    timeline = _extract_timeline_segments(result)
    if not timeline:
        return None

    best_seg: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for seg in timeline:
        score = _score_timeline_segment(seg, keywords)
        if score <= 0.0:
            continue
        if best_seg is None:
            best_seg = seg
            best_score = score
            continue
        seg_start = _safe_float(seg.get("start_time"), 0.0)
        best_start = _safe_float(best_seg.get("start_time"), 0.0)
        if score > best_score or (abs(score - best_score) < 1e-9 and seg_start < best_start):
            best_seg = seg
            best_score = score

    if best_seg is None:
        return None

    refined = _refine_with_word_timings(best_seg.get("words") or [], keywords)
    if refined is not None:
        refined["matched_text"] = str(best_seg.get("text") or "").strip()
        return refined

    start = _safe_float(best_seg.get("start_time"), 0.0)
    end = _safe_float(best_seg.get("end_time"), start)
    return {
        "start_time": start,
        "end_time": max(start, end),
        "matched_text": str(best_seg.get("text") or "").strip(),
    }


def _inject_primary_timestamp(text: str, exact_ts: str, language_hint: str) -> str:
    updated = re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", f"[{exact_ts}]", text, count=1)
    if updated == text:
        updated = re.sub(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", exact_ts, text, count=1)
    if exact_ts not in updated[:120]:
        prefix = {
            "ar": f"الطابع الزمني الأدق هو [{exact_ts}].\n",
            "fr": f"Le timestamp exact est [{exact_ts}].\n",
            "en": f"The exact timestamp is [{exact_ts}].\n",
        }.get(language_hint, f"The exact timestamp is [{exact_ts}].\n")
        updated = prefix + updated
    return updated


# ─────────────────────────────────────────────────────────────────────
# LANGUAGE / INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────

def _detect_language_hint(
        question: str,
        results: Optional[List[Any]] = None,
) -> str:
    """Detect the language the user WROTE their query in.

    Priority order:
      1. Explicit override keywords ("answer in arabic", "réponds en français", …)
      2. Script / marker detection on the query itself
      3. Default to English — NEVER fall through to the video's dominant language,
         because that caused English queries on Arabic videos to be answered in Arabic.
    """
    q = question or ""
    q_lower = q.lower()

    # ── 1. Explicit target-language override ─────────────────────────
    if any(kw in q_lower for kw in ("in arabic", "en arabe", "بالعربية", "إلى العربية", "to arabic")):
        return "ar"
    if any(kw in q_lower for kw in ("in french", "en français", "en francais", "بالفرنسية", "to french")):
        return "fr"
    if any(kw in q_lower for kw in ("in english", "en anglais", "بالإنجليزية", "to english")):
        return "en"

    # ── 2a. Arabic script detection ──────────────────────────────────
    arabic_chars = sum(1 for c in q if "\u0600" <= c <= "\u06FF")
    if arabic_chars > 0 and arabic_chars >= len(q.replace(" ", "")) * 0.15:
        return "ar"

    # ── 2b. French marker / accent detection ─────────────────────────
    q_lower_spaced = f" {q_lower} "
    french_markers = [
        " est-ce que ", " quelle ", " quelles ", " quels ", " comment ",
        " pourquoi ", " vidéo ", " qu'est-ce ", " qu'il ", " n'est ",
        " c'est ", " d'un ", " l'homme ", " dans la ", " parle ",
        " cette ", " quel ", " quoi ", " sur le ", " sur la ",
        " dans le ", " il y a ", " combien ", " après ", " avant ",
        " peut-on ", " est-il ", " sont-ils ", " ils ",
    ]
    if any(m in q_lower_spaced for m in french_markers):
        return "fr"
    french_accents = sum(1 for c in q if c in "àâäéèêëïîôùûüçœæÀÂÄÉÈÊËÏÎÔÙÛÜÇŒÆ")
    if french_accents >= 2:
        return "fr"

    # ── 3. Default: English ───────────────────────────────────────────
    # NOTE: We intentionally do NOT use the video's dominant language here.
    # The answer language must match the QUERY language, not the video language.
    # An English question about an Arabic video must receive an English answer.
    return "en"


# ─────────────────────────────────────────────────────────────────────
# TIMESTAMP EXTRACTION — detects "at 1:23", "in the min 1:00", etc.
# ─────────────────────────────────────────────────────────────────────

_TS_PATTERNS = [
    # explicit MM:SS or HH:MM:SS
    r"(?:at|around|in|@|vers|à|عند|في)\s+(\d{1,2}:\d{2}(?::\d{2})?)",
    r"\b(\d{1,2}:\d{2}:\d{2})\b",
    r"\b(\d{1,2}:\d{2})\b",
    # "at minute 1", "in the min 1:00", "at 1 minute"
    r"(?:min(?:ute)?s?|دقيقة|دقائق|minute)\s+(\d{1,2}(?::\d{2})?)",
    r"(\d{1,2})\s*(?:min(?:ute)?s?|mn)\b",
    # "at second 30", "at 30 seconds"
    r"(?:second(?:s)?|sec|ثانية)\s+(\d+)",
    r"(\d+)\s*(?:seconds?|sec)\b",
]

def _extract_all_timestamps(question: str) -> List[float]:
    """
    MULTI-TS-1: Extract ALL timestamps from a question as a sorted list of seconds.
    Handles questions like "what happened at 2:00 and at 5:30?" → [120.0, 330.0]
    Returns empty list if none found.
    """
    q = question.strip()
    found: List[float] = []
    seen_positions: List[Tuple[int, int]] = []   # track match spans to avoid double-counting

    for pat in _TS_PATTERNS:
        for m in re.finditer(pat, q, re.IGNORECASE):
            # Skip if this span overlaps one we already captured
            start_pos, end_pos = m.span()
            overlap = any(
                not (end_pos <= s or start_pos >= e)
                for s, e in seen_positions
            )
            if overlap:
                continue
            raw = m.group(1).strip()
            seconds: Optional[float] = None
            if ":" in raw:
                parts = raw.split(":")
                try:
                    parts_int = [int(p) for p in parts]
                    if len(parts_int) == 2:
                        seconds = float(parts_int[0] * 60 + parts_int[1])
                    elif len(parts_int) == 3:
                        seconds = float(parts_int[0] * 3600 + parts_int[1] * 60 + parts_int[2])
                except ValueError:
                    pass
            else:
                try:
                    val = float(raw)
                    if re.search(r"min(?:ute)?s?", pat, re.IGNORECASE):
                        seconds = val * 60.0
                    else:
                        seconds = val
                except ValueError:
                    pass
            if seconds is not None:
                found.append(seconds)
                seen_positions.append((start_pos, end_pos))

    # Deduplicate and sort
    unique = sorted(set(found))

    # Also check for Arabic ordinal minute references (e.g. "الدقيقة الثانية")
    ar_sec = _ar_ordinal_to_seconds(question)
    if ar_sec is not None and ar_sec not in unique:
        unique = sorted(unique + [ar_sec])

    return unique


def _extract_timestamp_seconds(question: str) -> Optional[float]:
    """
    Backward-compatible single-timestamp extractor.
    Returns the FIRST timestamp found, or None.
    """
    ts_list = _extract_all_timestamps(question)
    return ts_list[0] if ts_list else None


def _filter_results_by_timestamp(
    results: List[Any],
    target_seconds: float,
    window: float = 30.0,
) -> List[Any]:
    """
    Return results whose time range overlaps [target - window, target + window].
    Results are sorted by proximity to target_seconds (closest first).
    Falls back to full results if nothing found in window.
    """
    lo = max(0.0, target_seconds - window)
    hi = target_seconds + window
    matched = []
    for r in results:
        start = _safe_float(_get(r, "start_time") or _get(r, "start"), 0.0)
        end = _safe_float(_get(r, "end_time") or _get(r, "end"), start + 30.0)
        if start <= hi and end >= lo:
            matched.append((abs(start - target_seconds), r))
    if matched:
        matched.sort(key=lambda x: x[0])   # closest to target first
        return [r for _, r in matched]
    return results


# ═════════════════════════════════════════════════════════════════════
# LVBENCH INTENT LAYER  (added for LVBench / RAGY evaluation)
# ---------------------------------------------------------------------
# Maps questions to the 6 real LVBench question types so routing, modality,
# retrieval depth and the system prompt all line up with the benchmark.
#
#   entity recognition  → "counting" (how many) or falls through to factual
#   event understanding → "event"   (visual actions, usually NO timestamp)
#   key info retrieval  → "key_info" (exact on-screen text / numbers)
#   temporal grounding  → "temporal_grounding" (explicit MM:SS in query)
#   reasoning           → "reasoning" (why / cause→effect, multi-hop)
#   summarization       → "summarization" (MCQ over an event sequence)
#
# KEY FACT: most counting/event questions have NO timestamp, so we DON'T
# rely on time windows — we force visual modality + higher top_k instead.
# ═════════════════════════════════════════════════════════════════════

_LV_COUNTING = re.compile(
    r"\bhow\s+many\b|\bnumber\s+of\b|\bhow\s+(?:much|often)\b|\bcount\b"
    r"|\bcombien\b|كم\s|عدد\s",
    re.IGNORECASE,
)
_LV_KEY_INFO = re.compile(
    r"\bwhat\s+(?:year|number|word|date|time|name|caption|title|score)\b"
    r"|\bfirst\s+word\b|\blast\s+word\b|\bopening\s+caption\b|\bon[- ]?screen\b"
    r"|\bwritten\b|\bcaption\b"
    r"|\bwhat\s+(?:does|do)\s+(?:the\s+)?(?:sign|text|caption|subtitle|banner)\b"
    r"|quel(?:le)?\s+(?:année|numéro|mot|date)"
    r"|ما\s+(?:هي\s+)?(?:السنة|الكلمة|الرقم|التاريخ)",
    re.IGNORECASE,
)
_LV_TG_RANGE = re.compile(r"\b(\d{1,2}:\d{2})\b[^0-9]{0,8}\b(\d{1,2}:\d{2})\b")
_LV_TG_POINT = re.compile(r"\bat\s+(\d{1,2}:\d{2})\b", re.IGNORECASE)
_LV_TG_FROM = re.compile(r"\bfrom\s+\d{1,2}:\d{2}", re.IGNORECASE)
_LV_EVENT = re.compile(
    r"\bwhat\s+happens?\b|\bwhat\s+does\s+.+\s+do\b|\bwhat\s+(?:do|did)\s+.+\s+do\b"
    r"|\bwhen\s+.+\s+(?:does|do|did)\b"
    r"|\b(?:knock|hit|throw|kick|push|pull|grab|catch|jump|fall|run|walk|kneel|"
    r"stand|sit|enter|leave|open|close|break|cut|wipe|tie|approach|confront|"
    r"fight|kill|injure|stab|raise|lower|hold|wear|put|take|wave)\b"
    r"|\bwhat\s+(?:is|are)\s+.+\s+doing\b|\bhappen(?:s|ed|ing)?\s+(?:when|after|before)\b"
    r"|que\s+(?:se\s+passe|fait)|ماذا\s+(?:يحدث|يفعل|تفعل)",
    re.IGNORECASE,
)
_LV_REASONING = re.compile(
    r"\bwhy\b|\bbecause\b|\breason\b|\bcause\b|\bwhat\s+(?:makes|made)\b"
    r"|pourquoi|raison|لماذا|لأن|السبب",
    re.IGNORECASE,
)
_LV_SUMMARIZATION = re.compile(
    r"\bbest\s+summari[sz]es?\b|\bwhat\s+summari[sz]es?\b"
    r"|\bsummari[sz]e\s+the\s+(?:events?|sequence|first\s+half|second\s+half)\b"
    r"|\bsequence\s+of\s+events\b",
    re.IGNORECASE,
)

# MCQ option stripping — clean the question before EMBEDDING
_LV_MCQ_OPTION_LINE = re.compile(r"\n\s*\(?[A-D]\)?[\.\)]?\s.*", re.DOTALL)
_LV_MCQ_INLINE = re.compile(r"\s*\([A-D]\)\s.*$", re.DOTALL)

# Intents that are answered primarily from VISUAL frames, not speech.
_LV_VISUAL_INTENTS = {"counting", "event", "temporal_grounding", "key_info"}


def detect_lvbench_intent(question: str) -> Optional[str]:
    """Return an LVBench intent, or None if no LVBench pattern matches."""
    if not question:
        return None
    q = question
    if _LV_TG_RANGE.search(q) or _LV_TG_POINT.search(q) or _LV_TG_FROM.search(q):
        return "temporal_grounding"
    if _LV_SUMMARIZATION.search(q):
        return "summarization"
    if _LV_COUNTING.search(q):
        return "counting"
    if _LV_KEY_INFO.search(q):
        return "key_info"
    if _LV_REASONING.search(q):
        return "reasoning"
    if _LV_EVENT.search(q):
        return "event"
    return None


def lvbench_modality(intent: str) -> str:
    """Visual-dominant intents → 'visual'; everything else → 'both'."""
    if intent in ("counting", "event"):
        return "visual"
    return "both"


def lvbench_top_k(intent: str, default_top_k: Optional[int]) -> int:
    """Bump retrieval depth for intents that need many frames/segments."""
    base = default_top_k or 8
    bumps = {
        "summarization": 20, "event": 15, "counting": 15,
        "reasoning": 12, "temporal_grounding": 14, "key_info": 10,
    }
    return max(base, bumps.get(intent, base))


def strip_mcq_options(question: str) -> str:
    """Remove (A)/(B)/(C)/(D) options so the embedding query is just the question."""
    if not question:
        return question
    cut = _LV_MCQ_OPTION_LINE.split(question, maxsplit=1)[0]
    cut = _LV_MCQ_INLINE.split(cut, maxsplit=1)[0]
    return cut.strip()


# Per-intent system-prompt blocks appended in _build_system_prompt.
_LVBENCH_PROMPTS = {
    "counting": (
        "This is a COUNTING question (how many). Answer ONLY from the visual frames — "
        "speech transcripts cannot count objects on screen. Examine each frame and count "
        "the EXACT number of instances. Give the integer FIRST, then cite the frame [MM:SS]. "
        "Never give a range; pick your best exact integer. For 'how many times', count "
        "repeated actions across frames.\n"
    ),
    "event": (
        "This is an EVENT UNDERSTANDING question about a VISUAL action. Answer from the "
        "visual frames — the transcript says what was SAID, not what was SHOWN. Describe the "
        "specific action/sequence (who does what to whom, in what order) and cite the frame "
        "[MM:SS]. Small details (one hand vs two fingers, kills vs injures) decide the answer.\n"
    ),
    "key_info": (
        "This is a KEY INFORMATION question needing EXACT on-screen text or a precise fact "
        "(a year, number, first word, caption). Use OCR / on-screen text FIRST. Quote it "
        "EXACTLY — do not paraphrase or reorder digits (1636 ≠ 1633). For 'opening'/'first', "
        "focus on the earliest frames. Cite the [MM:SS] of the text.\n"
    ),
    "temporal_grounding": (
        "This is a TEMPORAL GROUNDING question about a SPECIFIC TIME WINDOW. Describe what "
        "HAPPENS (the visual event sequence) in that window in chronological order — not just "
        "what is said. Match the sequence against the options and pick the exact one.\n"
    ),
    "reasoning": (
        "This is a REASONING question (why / cause→effect). The cause and effect may appear at "
        "DIFFERENT moments. Think step by step: identify the effect, find the earlier cause, "
        "connect them, cite BOTH [MM:SS]. Then state which option matches.\n"
    ),
    "summarization": (
        "This is a SUMMARIZATION question over an EVENT SEQUENCE (not the whole video). The "
        "chunks are sorted chronologically. Reconstruct the sequence in order, then compare "
        "against the options — they are similar, so match EVERY detail (barefoot vs not, who "
        "stops whom, what is eaten/drunk).\n"
    ),
}


def _detect_intent(question: str) -> str:
    """
    Classify the question into one of:
      timestamp_lookup | translate | speaker | visual | yes_no |
      summary | temporal | list | factual

    Priority (highest → lowest):
      1. timestamp_lookup  — explicit MM:SS or Arabic ordinal minute
      2. translate         — translation request
      3. speaker           — who/how-many speaks
      4. visual            — appearance/clothing/color
      5. yes_no            — starts with yes/no auxiliary
      6. summary           — what is the video about
      7. temporal          — when did something happen
      8. list              — enumerate / how many (non-speaker)
      9. factual           — catch-all
    """
    q = question.strip()
    ql = q.lower()

    # 1. Explicit timestamp (MM:SS / HH:MM:SS) or Arabic ordinal minute
    if _extract_timestamp_seconds(question) is not None:
        return "timestamp_lookup"
    if _ar_ordinal_to_seconds(question) is not None:
        return "timestamp_lookup"

    # 2. Translation
    for p in TRANSLATE_PATTERNS:
        if re.search(p, ql):
            return "translate"

    # 2.5 LVBENCH intents — counting / event / key_info / temporal_grounding /
    #     reasoning / summarization. Checked before the generic visual/yes_no/
    #     list/factual fallbacks so LVBench questions route correctly. Returns
    #     None when no LVBench pattern fits, then we fall through to the
    #     original logic below.
    _lv = detect_lvbench_intent(question)
    if _lv is not None:
        return _lv

    # 3. Speaker identity / count — before list (كم) and factual (من)
    for p in SPEAKER_PATTERNS:
        if re.search(p, q):          # use original case for Arabic
            return "speaker"

    # 4. Visual appearance
    for p in VISUAL_PATTERNS:
        if re.search(p, ql):
            return "visual"

    # 5. Yes/No — anchored at start of question
    for p in YES_NO_PATTERNS:
        if re.search(p, ql):
            return "yes_no"

    # 6. Summary — what is the video about
    for p in SUMMARY_PATTERNS:
        if re.search(p, q):          # use original case for Arabic
            return "summary"

    # 7. Temporal — when / at what point
    for p in TEMPORAL_PATTERNS:
        if re.search(p, q):
            return "temporal"

    # 8. List — enumerate
    for p in LIST_PATTERNS:
        if re.search(p, q):
            return "list"

    # 9. Factual — catch-all
    for p in FACTUAL_PATTERNS:
        if re.search(p, ql):
            return "factual"

    return "factual"


# ─────────────────────────────────────────────────────────────────────
# QUERY MODALITY CLASSIFIER
# Determines whether the user needs speech/audio evidence, visual/frame
# evidence, or both — BEFORE the vector search runs.
#
# Strategy (fast → slow, stops at first confident signal):
#   1. Hard visual signals   → "visual"   (regex, 0 ms)
#   2. Hard speech signals   → "speech"   (regex, 0 ms)
#   3. Ambiguous / mixed     → LLM call   (~200 ms, only when needed)
#   4. LLM unavailable       → "both"     (safe fallback)
#
# Return values:
#   "speech"  — only pull speech/transcript chunks
#   "visual"  — only pull visual/frame chunks
#   "both"    — pull both (default for factual/summary questions)
# ─────────────────────────────────────────────────────────────────────

# Patterns that are DEFINITELY visual — no ambiguity
_MODALITY_VISUAL_HARD = re.compile(
    r"\b(wear|wearing|wore|outfit|dress(?:ed)?|clothes|clothing|shirt|jacket|suit|"
    r"uniform|costume|appear(?:ance)?|looks?\s+like|hair|glasses|beard|"
    r"color|colour|red|blue|green|black|white|pink|yellow|purple|"
    r"image|photo|picture|frame|screen|slide|board|diagram|chart|graph|"
    r"on.?screen|background|banner|logo|sign|text\s+on|written\s+on|"
    r"show(?:ing|s|n)?|display(?:ing|s|ed)?|visible|see\b|"
    # Flags, emblems, national symbols — visual by definition
    r"flag|flags|banner|banners|emblem|emblems|symbol|symbols|"
    r"coat\s+of\s+arms|national\s+symbol|insignia|"
    # Arabic visual
    r"ملابس|يلبس|يرتدي|مظهر|صورة|شاشة|لوحة|شعار|نص\s+على|"
    r"علم|أعلام|راية|رايات|شعار\s+الدولة|رمز|رموز|"   # flag/banner/emblem in Arabic
    # French visual
    r"porte|vêtement|apparence|écran|tableau|affiché|visible|montre|"
    r"drapeau|drapeaux|emblème|symbole|fanion)",          # flag/emblem in French
    re.IGNORECASE,
)

# Patterns that are DEFINITELY speech/audio — no ambiguity
_MODALITY_SPEECH_HARD = re.compile(
    r"\b(say|said|says|mention(?:ed|s)?|talk(?:ed|s|ing)?|speak|spoke|spoken|"
    r"quote|stated|explain(?:ed|s)?|discuss(?:ed|es)?|describe(?:d|s)?|"
    r"words?|transcript|audio|voice|hear|heard|listening|narrat\w*|"
    r"told|tell|argue[ds]?|claim(?:ed|s)?|announce[ds]?|"
    # Arabic speech
    r"قال|يقول|ذكر|يذكر|تحدث|يتحدث|كلمات|شرح|وصف|ناقش|"
    # French speech
    r"dit|dire|mentionné|parle|explique|décrit|discute|annonce|affirme)",
    re.IGNORECASE,
)

# Questions that are clearly about content understanding — need both
_MODALITY_BOTH_SIGNALS = re.compile(
    r"\b(summar\w*|overview|explain|about|cover|topic|theme|what\s+happen\w*|"
    r"describe\s+the\s+video|tell\s+me\s+about|main\s+point|key\s+point|"
    r"résum\w*|sujet|موضوع|ملخص|فيديو)",
    re.IGNORECASE,
)

_MODALITY_LLM_SYSTEM = (
    "You classify user questions about a video into one of three search modalities.\n"
    "Reply with ONLY one word — no explanation, no punctuation:\n"
    "  speech  → the answer requires what was SAID/SPOKEN in the video\n"
    "  visual  → the answer requires what was SHOWN/SEEN on screen (images, frames, objects, people, text, slides)\n"
    "  both    → the answer requires BOTH spoken content AND visual frames\n\n"
    "KEY RULE: Any question about whether something APPEARS, EXISTS, IS SHOWN, IS VISIBLE, or IS PRESENT\n"
    "on screen is VISUAL — even if phrased as yes/no ('is there a flag?', 'do they show any slides?').\n"
    "Questions about objects, people's appearance, on-screen text, colors, logos, flags, charts,\n"
    "backgrounds, slides, signs, or anything you would answer by LOOKING at the video → visual.\n\n"
    "Examples:\n"
    "Q: what color is his tie? → visual\n"
    "Q: did he mention climate change? → speech\n"
    "Q: what did the speaker say at 3:20? → speech\n"
    "Q: what logo appears on the screen? → visual\n"
    "Q: is there a flag of UAE in the video? → visual\n"
    "Q: is there a flag shown? → visual\n"
    "Q: do they show any slides? → visual\n"
    "Q: is there a graph on screen? → visual\n"
    "Q: what is written on the board? → visual\n"
    "Q: what is the background? → visual\n"
    "Q: is anyone wearing a suit? → visual\n"
    "Q: are there any charts displayed? → visual\n"
    "Q: what objects are on the table? → visual\n"
    "Q: is there a screen showing data? → visual\n"
    "Q: what does the presenter look like? → visual\n"
    "Q: summarize the video → both\n"
    "Q: what topics are covered? → both\n"
    "Q: explain what was discussed and shown → both\n"
    "Q: ماذا قال عن الذكاء الاصطناعي؟ → speech\n"
    "Q: ما الذي يظهر على الشاشة؟ → visual\n"
    "Q: هل يوجد علم الإمارات في الفيديو؟ → visual\n"
    "Q: هل تظهر أي شرائح؟ → visual\n"
    "Q: ما هو خلفية المشهد؟ → visual\n"
    "Q: est-ce qu'il y a un drapeau? → visual\n"
    "Q: que montre l'écran? → visual\n"
)


def classify_query_modality(question: str, *, use_llm: bool = True) -> str:
    """
    Classify question into 'speech', 'visual', or 'both'.

    Fast regex path covers ~90% of queries with zero latency.
    LLM path handles edge cases like:
      - 'is the European flag shown while they discuss AI?'  → both
      - 'what flag appears on screen?'  → visual
      - 'did they show any slides about memory?'  → visual (not speech)

    Args:
        question:  the raw user query
        use_llm:   set False to skip LLM and return regex result only
                   (useful in tests or when Ollama is known to be slow)

    Returns:
        'speech' | 'visual' | 'both'
    """
    q = (question or "").strip()
    if not q:
        return "both"

    has_visual = bool(_MODALITY_VISUAL_HARD.search(q))
    has_speech = bool(_MODALITY_SPEECH_HARD.search(q))

    # Clear-cut cases — no LLM needed
    if has_visual and not has_speech:
        logger.debug("modality=visual (regex hard-visual) q=%r", q[:60])
        return "visual"
    if has_speech and not has_visual:
        logger.debug("modality=speech (regex hard-speech) q=%r", q[:60])
        return "speech"
    if _MODALITY_BOTH_SIGNALS.search(q) or (has_visual and has_speech):
        logger.debug("modality=both (regex both-signals) q=%r", q[:60])
        return "both"

    # Short factual questions with no modality signal — default to both
    # but give LLM a chance to be smarter
    if not use_llm:
        return "both"

    # LLM classification for ambiguous cases
    try:
        answer = chat_plain(
            prompt=f"Q: {q}",
            system=_MODALITY_LLM_SYSTEM,
            num_predict=5,       # only needs one word
            temperature=0.0,
            timeout=int(_cfg("ollama_modality_timeout", 30)),         # hard timeout — never block search
        ).strip().lower()
        # Normalise the LLM response
        if answer.startswith("visual"):
            result = "visual"
        elif answer.startswith("speech"):
            result = "speech"
        else:
            result = "both"
        logger.debug("modality=%s (LLM) q=%r ans=%r", result, q[:60], answer)
        return result
    except Exception as _e:
        logger.debug("modality LLM failed (%s) → both", _e)
        return "both"


# ─────────────────────────────────────────────────────────────────────
# CITATION BUILDER
# ─────────────────────────────────────────────────────────────────────

def _make_citation(result: Any, question: Optional[str] = None) -> Dict[str, Any]:
    chunk_start = _get(result, "start_time", _get(result, "start", None))
    chunk_end = _get(result, "end_time", _get(result, "end", None))
    start = chunk_start
    end = chunk_end
    meta = _get(result, "metadata", {}) or {}
    chunk_type = _get(result, "chunk_type", "speech") or "speech"

    precise_span = None
    if question and chunk_type == "speech" and _detect_intent(question) in ("temporal", "timestamp_lookup"):
        precise_span = _find_precise_span(result, question)
        if precise_span is not None:
            start = precise_span.get("start_time", chunk_start)
            end = precise_span.get("end_time", chunk_end)

    citation = {
        "chunk_id": _get(result, "chunk_id"),
        "video_id": _get(result, "video_id"),
        "language": _get(result, "language"),
        "speaker": _get(result, "speaker"),
        "start_time": start,
        "end_time": end,
        "start_ts": _format_ts(start),
        "end_ts": _format_ts(end),
        "chunk_start_time": chunk_start,
        "chunk_end_time": chunk_end,
        "chunk_start_ts": _format_ts(chunk_start),
        "chunk_end_ts": _format_ts(chunk_end),
        "key_points": _collect_key_points(result),
        "score": _score_of(result),
        "text": _result_text(result),
        "chunk_type": chunk_type,
        "content_type": meta.get("content_type"),
        "frame_path": meta.get("frame_path"),
    }
    if precise_span is not None:
        citation["matched_text"] = precise_span.get("matched_text") or citation["text"]
        citation["matched_word"] = precise_span.get("matched_word")
        citation["timestamp_source"] = "timeline_refined"
    return citation


# ─────────────────────────────────────────────────────────────────────
# CONTEXT COMPRESSION
# ─────────────────────────────────────────────────────────────────────

def _normalize_relevance_score(result: Any) -> float:
    score = _score_of(result)
    chunk_type = _get(result, "chunk_type", "speech") or "speech"
    if chunk_type == "visual" and -1.0 <= score <= 1.0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))



# ─────────────────────────────────────────────────────────────────────
# NOISE FILTER: drop ads, sponsors, intros before they reach the LLM
# ─────────────────────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    # Channel intros / outros only — these are never content
    r"don't forget to (like|subscribe|hit the bell)",
    r"subscribe (to|for)",
    r"hit the (like|bell|subscribe)",
    r"smash (the )?like",
    r"see you (in the next|next time|tomorrow)",
    r"thanks? for watching",
    r"thanks? for tuning in",
]

# Patterns that signal the chunk IS the content the user may be asking about.
# If the question overlaps with these, we must NOT filter the chunk — the user
# is specifically asking about the guide / sponsored segment.
_CONTENT_QUESTION_SIGNALS = re.compile(
    r"(guide|cash|money|earn|income|monetize|monetise|agent skill|ai skill|"
    r"roi|value.based|pricing|automation|sponsor|ad\b|advertis)",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(
    "|".join(_NOISE_PATTERNS),
    re.IGNORECASE,
)

# Patterns that make a chunk noise ONLY when the question is not about them
_SOFT_NOISE_PATTERNS = re.compile(
    r"(cold hard cash|make money|monetize|turn .{0,30} into (cash|money|income)|"
    r"earn .{0,30} per (month|week|day)|income stream|"
    r"this video is sponsored|sponsored by|brought to you by|"
    r"use code .{2,20} for|discount code|promo code|affiliate link|"
    r"link in (the )?description|link in (the )?bio|"
    r"hubspot|skillshare|squarespace|nord ?vpn|expressvpn|betterhelp|"
    r"curiosity ?stream|brilliant\.org|"
    r"(this|the) guide (is|covers|shows|lays out)|"
    r"lays out (five|5|four|4|three|3|six|6) (real )?paths|"
    r"five real paths|the title undersells|tactical(ly)?|"
    r"welcome to (my channel|the channel|silicon valley|this podcast|our podcast)|"
    r"hello[,\s]+ everyone[,\s]+(welcome|today))",
    re.IGNORECASE,
)

def _is_noise_chunk(result: Any, question: Optional[str] = None) -> bool:
    """Return True if this chunk is noise (channel intro/outro/ad) that the user
    is NOT asking about.

    Hard noise (like/subscribe CTAs) is always filtered.
    Soft noise (sponsor segments, money guides) is filtered ONLY when the
    question does not signal the user is asking about that content.
    """
    text = (_result_text(result) or "").strip()
    if not text:
        return False
    # Always drop hard noise (like/subscribe CTAs — never relevant)
    if _NOISE_RE.search(text):
        return True
    # Soft noise: only drop if the question is NOT asking about it
    if _SOFT_NOISE_PATTERNS.search(text):
        q = (question or "").strip()
        if q and _CONTENT_QUESTION_SIGNALS.search(q):
            # The user IS asking about this segment — keep it
            return False
        return True
    return False

def _proximity_boosted_score(result: Any, target_seconds: float) -> float:
    """
    PROX-1: Blend relevance score with a proximity bonus so chunks
    physically close to the requested timestamp always outrank distant
    chunks — even if the distant chunk has better keyword overlap.

    Formula:
        final = relevance * RELEVANCE_WEIGHT + proximity * PROXIMITY_WEIGHT

    proximity = 1.0 when the chunk contains the exact second,
                decays to 0.0 at PROXIMITY_HALF_LIFE seconds away.
    """
    RELEVANCE_WEIGHT  = 0.20
    PROXIMITY_WEIGHT  = 0.80
    PROXIMITY_HALF_LIFE = 10.0   # seconds at which proximity bonus = 0.5 (was 20 s)

    relevance = _normalize_relevance_score(result)

    start = _safe_float(_get(result, "start_time") or _get(result, "start"), 0.0)
    end   = _safe_float(_get(result, "end_time")   or _get(result, "end"),   start + 30.0)

    # Distance = 0 if target falls inside the chunk, else gap to nearest edge
    if start <= target_seconds <= end:
        distance = 0.0
    else:
        distance = min(abs(target_seconds - start), abs(target_seconds - end))

    proximity = 1.0 / (1.0 + distance / PROXIMITY_HALF_LIFE)

    return relevance * RELEVANCE_WEIGHT + proximity * PROXIMITY_WEIGHT


def _select_results_for_prompt(
        results: List[Any],
        *,
        question: Optional[str] = None,
        max_items: Optional[int] = None,
) -> List[Any]:
    if not results:
        return []

    intent = _detect_intent(question or "")

    # PROX-1: For timestamp_lookup, extract ALL requested timestamps
    # and pre-filter + re-rank using proximity-boosted scores.
    all_target_ts = _extract_all_timestamps(question or "") if intent == "timestamp_lookup" else []

    if intent == "timestamp_lookup" and all_target_ts:
        # Window filter: keep only chunks within 20 s of ANY requested timestamp.
        # Tight window = only what was actually said at that moment, not nearby topics.
        # We try 20 s first; if empty (sparse transcript), we widen to 45 s as fallback.
        windowed: List[Any] = []
        seen_ids: set = set()
        for window_size in (20.0, 45.0):
            for ts in all_target_ts:
                for r in _filter_results_by_timestamp(results, ts, window=window_size):
                    cid = _get(r, "chunk_id") or id(r)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        windowed.append(r)
            if windowed:
                break   # found chunks at this window — stop widening
        # If still nothing found, keep original results but log a warning
        if windowed:
            results = windowed
        else:
            logger.warning(
                "timestamp_lookup: no chunks found near %s even at 45 s — returning all results",
                all_target_ts,
            )

        # Re-rank by proximity-boosted score relative to the NEAREST requested timestamp
        def _best_proximity_score(r: Any) -> float:
            return max(_proximity_boosted_score(r, ts) for ts in all_target_ts)

        ranked = sorted(results, key=_best_proximity_score, reverse=True)

        # For timestamp_lookup: hard cap at 3 chunks per timestamp —
        # tight and precise. We show the closest chunk + immediate neighbours only.
        hard_cap = min(max_items if max_items is not None else 3, 3 * len(all_target_ts))

    else:
        # FIX-A: For non-timestamp intents, pre-filter by single target if present
        if intent == "timestamp_lookup":
            target_sec = _extract_timestamp_seconds(question or "")
            if target_sec is not None:
                windowed2 = _filter_results_by_timestamp(results, target_sec, window=60.0)
                results = windowed2 if windowed2 else results

        ranked = sorted(results, key=_score_of, reverse=True)
        hard_cap = max_items if max_items is not None else 12

    # FIX-B: Context-aware noise filter
    clean = [r for r in ranked if not _is_noise_chunk(r, question=question)]
    ranked = clean if clean else ranked

    q = (question or "").lower()
    # VISUAL-FIX: align this keyword set with _MODALITY_VISUAL_HARD so that
    # any query the modality classifier routes to visual also triggers
    # visual-first result selection here. Previously the two lists diverged,
    # causing flag/emblem/symbol queries to retrieve visual chunks but then
    # silently demote them in the LLM prompt.
    wants_visual = bool(_MODALITY_VISUAL_HARD.search(question or ""))

    # For timestamp_lookup with speech question: drop visual chunks entirely —
    # the user asked what was SAID, not what was shown.
    if intent == "timestamp_lookup" and re.search(
        r"\b(say|said|says|talk|talked|talks|talking|mention|mentioned|speak|spoke|words?|quote|discuss|discussed|about)\b",
        q,
    ):
        speech_only = [r for r in ranked if (_get(r, "chunk_type") or "speech") != "visual"]
        ranked = speech_only if speech_only else ranked

    selected: List[Any] = []
    seen_keys: set = set()

    if intent == "timestamp_lookup":
        # Simple top-N selection: proximity score already did the ranking
        for item in ranked:
            if len(selected) >= hard_cap:
                break
            # FIX-8: use content-fingerprint so two chunks with different IDs
            # but the same first 120 chars don't both make it into the prompt.
            _txt = (_result_text(item) or "")[:120].lower()
            import hashlib as _hl
            _fp = _hl.md5(_txt.encode("utf-8", errors="ignore")).hexdigest()[:12] if _txt else ""
            key = (_get(item, "chunk_id") or _fp, _fp)
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(item)
    else:
        normalized_scores = [_normalize_relevance_score(r) for r in ranked]
        top_score = normalized_scores[0] if normalized_scores else 0.0

        # FIX-8: yes_no questions need a much lower score floor so that weak but
        # relevant evidence (the one chunk that briefly mentions the queried entity)
        # is not silently dropped when the top chunk scores very high.
        if intent == "yes_no":
            floor = 0.15
            min_keep = min(8, len(ranked))
        else:
            # Lower floor so mid-score chunks (e.g. later-in-video job discussion)
            # are not dropped when the top chunk happens to score very high.
            floor = max(0.28, top_score * 0.55)
            # Always keep at least 5 chunks — broad questions need more evidence.
            min_keep = min(5, len(ranked))

        for idx, item in enumerate(ranked):
            _txt = (_result_text(item) or "")[:120].lower()
            import hashlib as _hl
            _fp = _hl.md5(_txt.encode("utf-8", errors="ignore")).hexdigest()[:12] if _txt else ""
            key = (_get(item, "chunk_id") or _fp, _fp)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            score = normalized_scores[idx]
            previous = normalized_scores[idx - 1] if idx > 0 else score
            drop = previous - score

            if len(selected) < min_keep:
                selected.append(item)
                continue
            if len(selected) >= hard_cap:
                break
            # Relax drop threshold: 0.30 (was 0.22) so a single weak chunk
            # does not terminate the selection chain prematurely.
            if score < floor or drop > 0.30:
                break
            selected.append(item)

        # FIX-VISUAL-2: Always inject visual frames when they exist — not only when
        # wants_visual is True. For visual-specific questions, keep the old behaviour
        # (visual-first). For all other questions, append up to 2 frames at the end
        # so the frontend chunk panel always shows frames alongside audio evidence.
        visual_items   = [r for r in ranked if _get(r, "chunk_type") == "visual"]
        audio_items    = [r for r in selected if (_get(r, "chunk_type") or "speech") != "visual"]
        already_visual = [r for r in selected if _get(r, "chunk_type") == "visual"]

        def _vkey(item):
            _t = (_result_text(item) or "")[:120].lower()
            import hashlib as _hl
            _fp = _hl.md5(_t.encode("utf-8", errors="ignore")).hexdigest()[:12] if _t else ""
            return (_get(item, "chunk_id") or _fp, _fp)

        if visual_items:
            if wants_visual:
                # Visual-first layout: inject frames at the front, keep 3 audio
                if not already_visual:
                    for item in visual_items[:4]:
                        key = _vkey(item)
                        if key not in seen_keys and len(selected) < hard_cap:
                            selected.insert(0, item)
                            seen_keys.add(key)
                    selected = [r for r in selected if _get(r, "chunk_type") == "visual"] + audio_items[:3]
            else:
                # Non-visual question: quietly append up to 2 visual frames at the end
                # so the frontend can render them without displacing audio context.
                if not already_visual:
                    for item in visual_items[:2]:
                        key = _vkey(item)
                        if key not in seen_keys and len(selected) < hard_cap:
                            selected.append(item)
                            seen_keys.add(key)

    final = selected or ranked[: min(len(ranked), hard_cap)]

    # Re-sort chronologically so the LLM reads segments in video order
    intent = _detect_intent(question or "")
    if intent not in ("temporal",):
        def _start_of(item):
            t = _get(item, "start_time") or _get(item, "start") or 0
            try:
                return float(t)
            except (TypeError, ValueError):
                return 0.0
        final = sorted(final, key=_start_of)

    return final


def _select_citations_for_output(
        citations: List[Dict[str, Any]],
        question: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """BUG-5 FIX: Return citations sorted chronologically for all intents.

    Previous behaviour:
    - temporal  → returned only the single best-scored citation (lost all others)
    - all other → returned citations in score order (wrong video order)

    New behaviour:
    - temporal  → best-scored citation first (for the primary timestamp badge),
                  then remaining sorted chronologically (full evidence preserved)
    - all other → all citations sorted chronologically by start_time
    """
    if not citations:
        return []

    def _start_ts(c: Dict[str, Any]) -> float:
        return _safe_float(c.get("start_time"), 10**9)

    def _normalized_score(c: Dict[str, Any]) -> float:
        score = _safe_float(c.get("score"), 0.0)
        chunk_type = c.get("chunk_type") or "speech"
        if chunk_type == "visual" and -1.0 <= score <= 1.0:
            return max(0.0, min(1.0, (score + 1.0) / 2.0))
        if 0.0 <= score <= 1.0:
            return score
        return 1.0 / (1.0 + math.exp(-score))

    intent = _detect_intent(question or "")

    if intent in ("temporal", "timestamp_lookup"):
        # Keep the highest-precision citation first (primary timestamp badge),
        # then append the rest in chronological order so the UI can show a
        # full timeline if it wants to.
        valid = [
            c for c in citations
            if c.get("start_time") is not None or c.get("start_ts")
        ] or list(citations)

        refined = [c for c in valid if c.get("timestamp_source") == "timeline_refined"]
        pool = refined or valid

        def _rank_key(c: Dict[str, Any]):
            exact = 1 if c.get("timestamp_source") == "timeline_refined" else 0
            non_visual = 1 if (c.get("chunk_type") or "speech") != "visual" else 0
            return (exact, non_visual, _normalized_score(c), -_start_ts(c))

        best = sorted(pool, key=_rank_key, reverse=True)[0]
        rest = sorted([c for c in pool if c is not best], key=_start_ts)
        return [best] + rest

    # For summary / translate / list / factual / yes_no:
    # sort all citations chronologically so the answer narrative matches
    # the video order.
    return sorted(citations, key=_start_ts)


def _compress_results_for_prompt(
        results: List[Any],
        max_items: int = 8,
        question: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    ranked = _select_results_for_prompt(results, question=question, max_items=max_items)
    citations = [_make_citation(r, question=question) for r in ranked]
    output_citations = _select_citations_for_output(citations, question=question)
    blocks = []
    for i, c in enumerate(citations, start=1):
        kp = c.get("key_points") or []
        kp_lines = "\n".join([f"    • {x}" for x in kp[:3]]) if kp else ""
        speaker = c.get("speaker") or ""
        speaker_line = ""
        if speaker and speaker not in ("unknown", "SPEAKER_00", ""):
            speaker_line = f"  Speaker: {speaker}\n"
        lang = c.get("language") or ""
        lang_line = ""
        if lang and lang != "unknown":
            lang_line = f"  Language: {lang}\n"
        chunk_type = c.get("chunk_type") or "speech"
        type_line = "  Type: visual frame\n" if chunk_type == "visual" else ""
        raw_text = (c.get("matched_text") or c.get("text") or "").strip()
        # Strip raw JSON/metadata leakage from visual chunk text before sending to LLM
        if chunk_type == "visual" and raw_text:
            raw_text = _clean_visual_text_for_llm(raw_text)
        text = raw_text
        if len(text) > 320:
            text = text[:317] + "..."
        block = (
            f"[EXCERPT {i}] [{c.get('start_ts', '??:??')} → {c.get('end_ts', '??:??')}]\n"
            f"{speaker_line}{lang_line}{type_line}"
            f"  \"{text}\"\n"
        )
        if kp_lines:
            block += f"  Key points:\n{kp_lines}\n"
        blocks.append(block)
    return "\n".join(blocks), output_citations


# ─────────────────────────────────────────────────────────────────────
# VIDEO-AWARE-2: Video context block builder
# ─────────────────────────────────────────────────────────────────────

def _build_video_context_block(
        video_metadata: Optional[Dict[str, Any]],
        language_hint: str,
) -> str:
    """
    VIDEO-AWARE-2: Format the video's high-level overview into a prompt block.
    This gives the LLM the "big picture" before reading individual chunks,
    dramatically improving answer coherence on vague or broad questions.
    """
    if not video_metadata:
        return ""

    title = (video_metadata.get("title") or "").strip()
    summary = (video_metadata.get("summary") or "").strip()
    keywords = video_metadata.get("keywords") or []
    topics = video_metadata.get("topics") or []
    domain = (video_metadata.get("domain") or "").strip()
    language = (video_metadata.get("language") or "").strip()
    duration = video_metadata.get("duration_seconds")

    if not (title or summary):
        return ""

    # Truncate summary to 500 chars to save tokens
    if len(summary) > 500:
        summary = summary[:497] + "..."

    if language_hint == "ar":
        header = "═══ نظرة عامة على الفيديو ═══"
        title_label = "العنوان"
        summary_label = "الملخص"
        topics_label = "المواضيع"
        domain_label = "المجال"
        duration_label = "المدة"
    elif language_hint == "fr":
        header = "═══ Aperçu de la vidéo ═══"
        title_label = "Titre"
        summary_label = "Résumé"
        topics_label = "Sujets"
        domain_label = "Domaine"
        duration_label = "Durée"
    else:
        header = "═══ Video Overview ═══"
        title_label = "Title"
        summary_label = "Summary"
        topics_label = "Topics"
        domain_label = "Domain"
        duration_label = "Duration"

    lines = [header]
    if title:
        lines.append(f"{title_label}: {title}")
    if domain:
        lines.append(f"{domain_label}: {domain}")
    if duration:
        dur_fmt = _format_ts(duration)
        lines.append(f"{duration_label}: {dur_fmt}")
    if summary:
        lines.append(f"{summary_label}: {summary}")
    if topics:
        lines.append(f"{topics_label}: {', '.join(topics[:8])}")
    if keywords:
        kw_str = ", ".join(keywords[:10])
        lines.append(f"Keywords: {kw_str}")

    # Inject chapters so the LLM knows what topics are covered and when
    chapters = video_metadata.get("chapters") or []
    if chapters:
        if language_hint == "ar":
            lines.append("الفصول:")
        elif language_hint == "fr":
            lines.append("Chapitres:")
        else:
            lines.append("Chapters:")
        for ch in chapters[:30]:
            ch_start = ch.get("start_time") or ch.get("start") or 0
            ch_title = (ch.get("title") or ch.get("label") or "").strip()
            if ch_title:
                lines.append(f"  [{_format_ts(ch_start)}] {ch_title}")

    lines.append("═" * 30)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CONF-1: Confidence scoring
# ─────────────────────────────────────────────────────────────────────

def _confidence_score(citations: List[Dict[str, Any]]) -> float:
    """
    CONF-1: Compute a 0-1 confidence score from retrieved citations.

    Rules:
    - No citations → 0.0
    - Score is weighted average of top-3 citation scores, with a
      bonus for having multiple strong citations.
    - Visual-only results are penalised (they answer fewer question types).
    - A single weak citation caps confidence at 0.4.
    """
    if not citations:
        return 0.0

    scores: List[float] = []
    for c in citations[:3]:
        raw = _safe_float(c.get("score"), 0.0)
        chunk_type = c.get("chunk_type") or "speech"
        # Normalise CLIP cosine scores from [-1,1] to [0,1]
        if chunk_type == "visual" and -1.0 <= raw <= 1.0:
            raw = max(0.0, (raw + 1.0) / 2.0)
        # Normalise logistic for out-of-range scores
        if raw > 1.0 or raw < 0.0:
            raw = 1.0 / (1.0 + math.exp(-raw))
        # Penalise visual-only slightly
        if chunk_type == "visual":
            raw *= 0.85
        scores.append(raw)

    if not scores:
        return 0.0

    # Weighted average: top citation counts double
    weights = [2.0] + [1.0] * (len(scores) - 1)
    conf = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # Bonus when multiple citations agree
    if len(citations) >= 3:
        conf = min(1.0, conf * 1.1)

    # Single weak citation → cap
    if len(citations) == 1 and conf < 0.5:
        conf = min(conf, 0.4)

    return round(conf, 4)


# ─────────────────────────────────────────────────────────────────────
# CONV-2: Conversation history → Ollama message list
# ─────────────────────────────────────────────────────────────────────

def _build_history_messages(
        history: Optional[List["ConversationTurn"]],
        max_turns: int = 3,
) -> List[Dict[str, str]]:
    """
    CONV-3: Convert the last N ConversationTurn entries into Ollama
    message dicts that are injected between the system prompt and the
    current user message.

    Keeps answers short (≤400 chars) to stay within num_ctx budget.
    """
    if not history:
        return []
    messages: List[Dict[str, str]] = []
    for turn in history[-max_turns:]:
        q = (turn.question or "").strip()
        a = (turn.answer or "").strip()
        if not q or not a:
            continue
        # Truncate long prior answers to save context tokens
        if len(a) > 400:
            a = a[:397] + "…"
        messages.append({"role": "user",      "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages




_SIMPLE_QUESTION_RE = re.compile(
    r"^(what|who|when|where|did|does|is|was|were|how|why|can|could|would|should)\b"
    r".{0,80}(min(ute)?s?\s*\d+|\d+\s*min(ute)?s?|at\s+\d|around\s+\d|timestamp|moment|point|mention|say|talk|speak|discuss).*$",
    re.IGNORECASE,
)

_BROAD_QUESTION_RE = re.compile(
    r"(summar|overview|explain|describe|tell me about|what is the (main|key|overall)|"
    r"what does .{0,30} cover|what are the (main|key|top)|give me a|walk me through)",
    re.IGNORECASE,
)

def _question_complexity(question: str) -> str:
    """Return 'simple', 'broad', or 'normal'."""
    q = (question or "").strip()
    if _BROAD_QUESTION_RE.search(q):
        return "broad"
    if len(q) < 80 and _SIMPLE_QUESTION_RE.match(q):
        return "simple"
    if len(q) < 60:
        return "simple"
    return "normal"

def _length_instruction(question: str, intent: str, language_hint: str) -> str:
    """Return an explicit length + opening style instruction for the LLM."""
    complexity = _question_complexity(question)

    if intent == "translate":
        return ""  # translation length = source length, don't constrain

    if intent in ("summary",) or complexity == "broad":
        instructions = {
            "ar": "اكتب إجابة شاملة — ٣ إلى ٥ جمل. ابدأ مباشرة بوصف محتوى الفيديو. لا تقل 'حول الدقيقة كذا'.",
            "fr": "Réponds en 3 à 5 phrases. Commence directement par le contenu, pas par 'Aux alentours de...'.",
            "en": "Answer in 3-5 sentences. Start directly with the content.",
        }
        return instructions.get(language_hint, instructions["en"])

    if intent == "list":
        instructions = {
            "ar": "أعطِ قائمة مرقمة. كل بند يجب أن يتضمن الطابع الزمني [MM:SS]. اكتب بالعربية.",
            "fr": "Donne une liste numérotée. Chaque item doit inclure son horodatage [MM:SS].",
            "en": "Give a numbered list. Each item must include its timestamp [MM:SS].",
        }
        return instructions.get(language_hint, instructions["en"])

    if intent == "speaker":
        instructions = {
            "ar": "أجب مباشرة: كم عدد المتحدثين وما أسماؤهم إن وُجدت. جملتان فقط.",
            "fr": "Réponds directement: combien de personnes parlent et qui sont-elles. 2 phrases max.",
            "en": "Answer directly: how many speakers and who they are. 2 sentences max.",
        }
        return instructions.get(language_hint, instructions["en"])

    if complexity == "simple" or intent in ("timestamp_lookup", "factual", "yes_no", "temporal"):
        instructions = {
            "ar": "أجب في ٢ إلى ٣ جمل فقط. ابدأ بالمحتوى مباشرة. لا تبدأ بـ'حول الدقيقة'.",
            "fr": "Réponds en 2-3 phrases maximum. Commence par le contenu directement.",
            "en": "Answer in 2-3 sentences max. Be direct. Only open with YES/NO if it is a yes/no question.",
        }
        return instructions.get(language_hint, instructions["en"])

    # normal
    instructions = {
        "ar": "أجب بشكل طبيعي — لا تُطَوّل دون داعٍ.",
        "fr": "Réponds naturellement, sans t'étendre inutilement.",
        "en": "Answer naturally — don't over-explain.",
    }
    return instructions.get(language_hint, instructions["en"])

def _build_system_prompt(
        intent: str,
        language_hint: str,
        video_title: str = "",
        video_domain: str = "",
        question: str = "",
) -> str:
    if video_title:
        video_ref = f'the video "{video_title}"'
        if video_domain and video_domain.lower() not in ("other", "unknown", ""):
            video_ref += f" ({video_domain})"
    else:
        video_ref = "a video"

    # Core identity — conversational, not analytical report writer
    base = (
        "/nothink\n"
        f"You are a helpful assistant who has watched {video_ref} and can answer questions about it naturally.\n"
        "Speak like a knowledgeable friend — direct, warm, conversational.\n"
        "Ground rules:\n"
        "- Answer in plain flowing prose. No bullet points, no headers, no bold text unless asked.\n"
        "- Never open with 'Around the X-minute mark...' — dive straight into what was said or done.\n"
        "- Weave timestamps like [05:13] naturally into the sentence only when they genuinely help.\n"
        "- CRITICAL — VIDEO-ONLY RULE: Every fact, statistic, name, quote, and claim MUST come "
        "directly from the excerpts provided. NEVER use your pre-training knowledge to fill gaps "
        "or enrich the answer — even if you recognize the speaker or topic. "
        "If a fact is not in the excerpts, do not say it.\n"
        "- NAMES: Only use a person's name exactly as it appears in the excerpts. "
        "If no name appears in the excerpts, do not name anyone.\n"
        "- STATISTICS & NUMBERS: Never include any number, percentage, timeframe, or growth rate "
        "that does not appear word-for-word in the excerpts. This is the most common hallucination.\n"
        "- If the excerpts don't fully answer the question, say so honestly rather than "
        "supplementing from your own knowledge.\n"
        "- Quote directly only when the exact words matter; otherwise paraphrase naturally.\n"
        "- Sound human, not like a search result or a report.\n"
        "- NEVER contradict yourself within the same answer.\n"
        "- Be consistent: pick one clear interpretation of the evidence and stick with it.\n"
    )

    intent_hints = {
        "timestamp_lookup": (
            "The user is asking about a SPECIFIC MOMENT in the video. "
            "The excerpts provided have been pre-filtered to the time window around that exact moment — "
            "they are the ONLY source of truth you are allowed to use.\n"
            + (
                "The user wants to know what was SPOKEN or SAID at that moment. "
                "Your job: quote or closely paraphrase the ACTUAL words spoken in those excerpts. "
                "Lead with the timestamp [MM:SS] from the excerpt, then the words. "
                "Example format: 'At [01:20], the speaker says: \"...exact words...\"'\n"
                "STRICT RULES:\n"
                "- NEVER describe what the camera shows or the visual scene.\n"
                "- NEVER use content from other parts of the video — only the excerpts given.\n"
                "- If multiple excerpts are given, quote the one whose timestamp is closest to what the user asked.\n"
                "- If no excerpt covers that exact second, say: 'The closest I have is [MM:SS]: \"...\"' — always give something.\n"
                "- NEVER invent, paraphrase loosely, or substitute a thematically similar quote from elsewhere.\n"
                if re.search(
                    r"\b(say|said|says|talk|talked|talks|talking|mention|mentioned|speak|spoke|words?|quote|discuss|discussed|about"
                    r"|قال|يقول|ذكر|يذكر|تحدث|يتحدث|كلمات|اقتبس|ناقش)\b",
                    question.lower()
                )
                else
                "Describe what was being said or shown in those excerpts directly and concisely. "
                "Lead with the timestamp [MM:SS] from the excerpt. "
                "ONLY use the excerpts given — never pull in content from other parts of the video.\n"
            )
            + "ABSOLUTE RULE: Your answer timestamp MUST match one of the [MM:SS] timestamps in the excerpts above. "
            "Do NOT write a timestamp that does not appear in the excerpts.\n"
        ),
        "temporal": (
            "The user wants to know WHEN something happened in the video. "
            "Lead with the timestamp [MM:SS], then briefly describe what happened at that moment. "
            "If the event appears at multiple points, list all timestamps. "
            "If it cannot be pinpointed, give the closest moment you have.\n"
        ),
        "list": (
            "The user wants you to enumerate items from the video. "
            "Give a clean numbered list. "
            "Each item MUST include its timestamp [MM:SS] from the excerpts — place it right after the item. "
            "Example: '1. Cairo [00:12]  2. Alexandria [01:45]'. "
            "Never omit timestamps. Keep items concise. "
            "If the question is in Arabic, number the list with Arabic numerals and write in Arabic.\n"
        ),
        "visual": (
            "The user is asking about something VISUAL — what can be SEEN on screen.\n"
            "This includes: objects, people's appearance, on-screen text, slides, charts, graphs,\n"
            "flags, logos, backgrounds, signs, clothing, colors, or anything visible in the video.\n"
            "Answer ONLY from the visual frame descriptions provided in the 'Visual evidence' block.\n"
            "Do NOT rely on the speech/audio transcript to answer a visual question — the transcript\n"
            "describes what was SAID, not what was SHOWN.\n"
            "For yes/no visual questions (e.g. 'is there a flag?'): answer YES, NO, or CANNOT CONFIRM\n"
            "first, then cite the specific frame [MM:SS] and describe exactly what it shows.\n"
            "If no visual frames show the relevant detail, say clearly:\n"
            "'I cannot confirm this from the available video frames — the relevant moment may not\n"
            "have been captured as a keyframe.'\n"
            "Be specific: describe exactly what is visible and at what timestamp [MM:SS].\n"
        ),
        "yes_no": (
            "Answer YES, NO, or PARTIALLY first — one clear word — then explain with specifics from the excerpts. "
            "Include at least one timestamp [MM:SS] from the excerpts to anchor the key claim.\n"
            "CRITICAL RULE: Your verdict MUST be determined by the excerpts provided, not by your prior knowledge.\n"
            "If ANY excerpt contains evidence that the thing exists or was mentioned — even briefly — the answer is YES.\n"
            "IMPORTANT: If you cannot find evidence in the excerpts, do NOT claim the speaker never said it. "
            "Instead say: 'It does not appear in the retrieved segments — it may occur elsewhere in the video.'\n"
            "Once you say YES (or NO), do NOT contradict yourself in the same answer. Stay consistent throughout.\n"
            "Make the answer RICH and SPECIFIC: name the exact concepts, quote a key phrase, and describe their stance.\n"
            "If the question is in Arabic, answer YES as 'نعم', NO as 'لا', PARTIALLY as 'جزئياً'.\n"
        ),
        "speaker": (
            "The user wants to know how many speakers/people are talking in the video, or who they are.\n"
            "Count the distinct speaker labels or names that appear across the excerpts provided.\n"
            "Give a direct count first, then name the speakers if their names or labels are available.\n"
            "If speaker labels are generic (SPEAKER_00, SPEAKER_01 etc.), describe them by what they say "
            "or their role in the conversation (e.g. 'the interviewer', 'the guest').\n"
            "Do NOT invent names. Only use names that explicitly appear in the excerpts.\n"
            "If the question is in Arabic, answer entirely in Arabic.\n"
        ),
        "summary": (
            "The user wants to know what the video is about. "
            "Give a natural, flowing overview: what topic it covers, who speaks, what the key points are. "
            "Write it like you're telling a friend what the video is about — warm, direct, no bullet points unless asked. "
            "Draw on the Video Overview block (if present) AND the excerpts. "
            "Do NOT list timestamps unless they genuinely help. "
            "If the question is in Arabic, answer entirely in Arabic.\n"
        ),
        "translate": (
            "Translate the relevant passages faithfully into the target language. Don't summarise — translate the actual words.\n"
        ),
        "factual": (
            "Give a direct, specific answer grounded entirely in the excerpts. "
            "Weave in at least one timestamp [MM:SS] from the excerpts to anchor the claim — "
            "e.g. 'At [01:20], Yasser mentions that...'. Keep it concise but informative. "
            "If the question is in Arabic, answer entirely in Arabic.\n"
        ),
    }

    # LVBench intent prompt blocks override / extend the defaults above.
    intent_hints.update(_LVBENCH_PROMPTS)

    intent_block = intent_hints.get(intent, "Answer directly and naturally.\n")

    lang_block = {
        "ar": "أجب بالكامل باللغة العربية. اذكر الطوابع الزمنية بالشكل [MM:SS].\n",
        "fr": "Réponds entièrement en français. Cite les horodatages au format [MM:SS].\n",
        "en": "Respond entirely in English.\n",
        "es": "Responde completamente en español. Cita las marcas de tiempo como [MM:SS].\n",
        "de": "Antworte vollständig auf Deutsch. Zitiere Zeitstempel als [MM:SS].\n",
        "it": "Rispondi interamente in italiano. Cita i timestamp come [MM:SS].\n",
        "pt": "Responde inteiramente em português. Cite os timestamps como [MM:SS].\n",
    }.get(language_hint, "CRITICAL: Respond in the SAME language the user wrote their question in. Do NOT switch languages.\n")

    return base + intent_block + lang_block


def _build_user_prompt(
        question: str,
        compressed_context: str,
        intent: str,
        language_hint: str,
        graph_context: str = "",
        visual_context: str = "",
        video_context_block: str = "",      # VIDEO-AWARE-2
) -> str:
    preamble = {
        "ar": "المقاطع المستخرجة من الفيديو:",
        "fr": "Passages extraits de la vidéo :",
        "en": "Passages from the video:",
    }.get(language_hint, "Passages from the video:")

    # VIDEO-AWARE-2: prepend video overview block
    video_block = ""
    if video_context_block and video_context_block.strip():
        video_block = video_context_block.strip() + "\n\n"

    graph_block = ""
    if graph_context and graph_context.strip():
        gc_label = {
            "ar": "سياق الرسم المعرفي (كيانات / علاقات):",
            "fr": "Contexte du graphe de connaissances (entités / relations) :",
            "en": "Knowledge graph context (entities / relations):",
        }.get(language_hint, "Knowledge graph context (entities / relations):")
        graph_block = f"{gc_label}\n{graph_context.strip()}\n\n"

    visual_block = ""
    if visual_context and visual_context.strip():
        vc_label = {
            "ar": "الأدلة البصرية (تسميات إطارات المفتاح):",
            "fr": "Preuves visuelles (légendes des images clés) :",
            "en": "Visual evidence (keyframe captions):",
        }.get(language_hint, "Visual evidence (keyframe captions):")
        visual_block = (
            f"{vc_label}\n{visual_context.strip()}\n\n"
            "Each line is a key frame: caption, then any 'counts:', 'layout:', "
            "'objects:', 'board text:' or 'screen text:' extracted from that frame "
            "by the vision pipeline. For counting questions, use the 'counts:' "
            "values; for spatial / position questions ('what is behind / left of / "
            "next to X'), use 'layout:'; for on-screen text, use board/screen text. "
            "Treat these structured notes as reliable evidence.\n"
            "When visual evidence is relevant, cite timestamps in [MM:SS] format.\n\n"
        )

    # MULTI-TS-3: Build a note listing ALL timestamps the user asked about
    all_asked_ts = _extract_all_timestamps(question)
    coverage_note = ""
    if all_asked_ts:
        ts_labels = ", ".join(f"[{_format_ts(s)}]" for s in all_asked_ts)
        if len(all_asked_ts) == 1:
            coverage_note = (
                f"⚠️  TIMESTAMP LOCK: The user asked specifically about {ts_labels}. "
                "The excerpts below were retrieved for that exact window. "
                "You MUST base your answer ONLY on these excerpts. "
                "The timestamp you write in your answer MUST appear in the excerpt headers above (e.g. [01:20]). "
                "If you write any other timestamp, you are hallucinating — do not do it.\n\n"
            )
        else:
            coverage_note = (
                f"⚠️  TIMESTAMP LOCK: The user asked about MULTIPLE moments: {ts_labels}. "
                "Address EACH one separately. "
                "Every timestamp you write MUST come directly from the excerpt headers — never invent one.\n\n"
            )

    # Dynamic length + opening style instruction
    length_instruction = _length_instruction(question, intent, language_hint)

    return (
        f"Question: {question}\n\n"
        f"{video_block}"
        f"{graph_block}"
        f"{visual_block}"
        f"{coverage_note}"
        f"{preamble}\n\n"
        f"{compressed_context}\n\n"
        f"Instruction: {length_instruction}\n"
        f"FINAL REMINDER: Your answer must be grounded 100% in the excerpts above. "
        f"Do not add any fact, name, number, or claim that is not explicitly present in those excerpts. "
        f"If you are unsure whether something came from the excerpts or your own knowledge, leave it out.\n"
    )


# ─────────────────────────────────────────────────────────────────────
# ADAPTIVE PARAMETERS
# ─────────────────────────────────────────────────────────────────────

def _get_num_predict(intent: str, num_results: int) -> int:
    base = int(_cfg("ollama_num_predict", DEFAULT_NUM_PREDICT))
    # timestamp_lookup and factual should be concise — a short human answer,
    # not an essay. Only summary and translate genuinely need more tokens.
    multiplier = {
        "timestamp_lookup": 0.6,   # short conversational answer
        "factual":          0.6,
        "visual":           0.7,   # concise — describe what's seen
        "yes_no":           1.0,   # needs space for specific details
        "temporal":         0.8,
        "list":             1.2,
        "speaker":          0.7,   # direct count + names
        "summary":          1.5,
        "translate":        1.6,
    }.get(intent, 0.7)
    return max(128, min(1024, int(base * multiplier)))


def _get_temperature(intent: str) -> float:
    configured = float(_cfg("ollama_temperature", DEFAULT_TEMPERATURE))
    # Higher temperature = more natural, conversational language.
    # We keep factual/yes_no slightly lower for accuracy, but never 0.0
    # (that makes the LLM sound like a template filler).
    targets = {
        "yes_no":           0.2,
        "visual":           0.1,
        "factual":          0.15,
        "list":             0.2,
        "temporal":         0.2,
        "timestamp_lookup": 0.25,
        "speaker":          0.15,  # factual — count is precise
        "summary":          0.3,
        "translate":        0.1,
    }
    target = targets.get(intent, 0.2)
    return min(configured, target)


def _get_seed() -> int:
    return int(_cfg("ollama_seed", 42))


# ─────────────────────────────────────────────────────────────────────
# OLLAMA MODEL HANDLING
# ─────────────────────────────────────────────────────────────────────

def _parse_parameter_size_to_billions(value: Any) -> float:
    if not value:
        return 999.0
    s = str(value).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1))
    return 999.0


def _extract_model_family(model_name: str) -> str:
    if not model_name:
        return ""
    return model_name.split(":", 1)[0].strip().lower()


def _list_models() -> List[Dict[str, Any]]:
    base = _get_ollama_base_url()
    url = f"{base}/api/tags"
    try:
        r = requests.get(url, timeout=min(20, _timeout()))
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("Ollama /api/tags failed: %s", e)
        return []
    models = data.get("models", [])
    out: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        full_name = _normalize_text(m.get("name"))
        if not full_name:
            continue
        details = m.get("details") or {}
        parameter_size = details.get("parameter_size")
        size_billions = _parse_parameter_size_to_billions(parameter_size)
        family = _extract_model_family(full_name)
        out.append({"name": full_name, "family": family,
                    "size_billions": size_billions, "parameter_size": parameter_size})
    return out


def _choose_model(preferred_model: Optional[str] = None) -> Optional[str]:
    # If a model is explicitly configured, trust it and pass it directly to
    # Ollama without requiring it to appear in /api/tags (list may lag or
    # use a different tag format than the configured name).
    if preferred_model and preferred_model.strip():
        logger.debug("_choose_model: using configured model %r directly", preferred_model.strip())
        return preferred_model.strip()

    models = _list_models()
    if not models:
        logger.warning("No Ollama models found!")
        return None
    exact_names = {m["name"]: m for m in models}
    for candidate in PREFERRED_MODEL_PRIORITY:
        if candidate in exact_names:
            return candidate
    for family in PREFERRED_FAMILIES:
        family_models = [m for m in models if m["family"].startswith(family)]
        if family_models:
            family_models.sort(key=lambda x: (x["size_billions"], x["name"]))
            return family_models[0]["name"]
    models.sort(key=lambda x: (x["size_billions"], x["name"]))
    return models[0]["name"]


# ─────────────────────────────────────────────────────────────────────
# OLLAMA BLOCKING CALL
# ─────────────────────────────────────────────────────────────────────

def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _call_ollama_chat(
        model: str,
        system_prompt: str,
        user_prompt: str,
        intent: str,
        num_results: int,
        history_messages: Optional[List[Dict[str, str]]] = None,   # CONV-3
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    base = _get_ollama_base_url()
    url = f"{base}/api/chat"
    num_predict = _get_num_predict(intent, num_results)
    temperature = _get_temperature(intent)

    # CONV-3: inject history between system prompt and current user message
    messages = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 4096,
            "top_p": DEFAULT_TOP_P,
            "repeat_penalty": DEFAULT_REPEAT_PENALTY,
            "seed": _get_seed(),
        },
    }
    try:
        started = time.time()
        data = _post_json(url, payload, timeout_s=_timeout())
        elapsed = time.time() - started
        text = ""
        msg = data.get("message")
        if isinstance(msg, dict):
            text = (msg.get("content") or "").strip()
        if not text:
            text = (data.get("response") or "").strip()
        if text:
            logger.info("LLM OK | model=%s | intent=%s | tokens≈%d | %.1fs",
                        model, intent, len(text.split()), elapsed)
            text = _strip_thinking_tags(text)
        return text or None, data
    except requests.Timeout:
        logger.warning("Ollama TIMEOUT after %ds | model=%s", _timeout(), model)
        return None, None
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        logger.warning("Ollama HTTP %s | model=%s | %s", status, model, body)
        return None, None
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return None, None


def _strip_thinking_tags(text: str) -> str:
    """Remove Qwen3/3.5 <think>...</think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# STREAMING GENERATOR — v6.0 improved
# ─────────────────────────────────────────────────────────────────────

def stream_answer(
        question: str,
        results: List[Any],
        *,
        preferred_model: Optional[str] = None,
        language_hint: Optional[str] = None,
        graph_context: Optional[str] = None,
        graph_context_future: Optional[Any] = None,
        visual_context: Optional[str] = None,
        max_items: Optional[int] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,   # CONV-2
        modality: Optional[str] = None,   # VISUAL-FIX: 'speech'|'visual'|'both'
        benchmark: Optional[str] = None,  # BENCHMARK: 'video_mmlu' | 'eduvidqa' | None
) -> Generator[str, None, None]:
    """
    Streaming answer generator — v7.0.
    Adds: multi-timestamp merging, conversation history injection,
    confidence scoring in the final done event.

    VISUAL-FIX: mirrors synthesize_answer — accepts modality signal and
    auto-builds visual_context from the visual chunks in results.
    """
    import concurrent.futures as _cf
    results = results or []
    intent = _detect_intent(question)
    lang = language_hint or _detect_language_hint(question, results)

    # VISUAL-FIX: upgrade intent to "visual" when modality classifier says so.
    # LVBENCH: counting/event/key_info/temporal_grounding already carry their own
    # visual-aware prompts — default them to their LVBench modality and do NOT
    # collapse them back to the generic "visual" intent below.
    _lv_modality = lvbench_modality(intent) if intent in _LV_VISUAL_INTENTS else None
    _effective_modality = modality or _lv_modality or ("visual" if intent == "visual" else "both")
    _vis_chunks_in_results = [
        r for r in results if (_get(r, "chunk_type") or "speech") == "visual"
    ]
    has_visual_chunks = bool(_vis_chunks_in_results)

    if _effective_modality in ("visual", "both") and intent not in (
        "visual", "timestamp_lookup",
    ) and intent not in _LV_VISUAL_INTENTS and has_visual_chunks:
        # Same upgrade logic as synthesize_answer: visual chunks present + modality
        # says visual content is needed → use "visual" framing for the LLM prompt.
        if _effective_modality == "visual" or intent in ("yes_no", "factual", "list", "temporal"):
            intent = "visual"

    # VISUAL-FIX: auto-build visual_context from ALL visual chunks, sorted
    # chronologically, with a higher cap (12) for long-video coverage.
    if not visual_context and _vis_chunks_in_results:
        _vis_raw = sorted(
            [
                {
                    "timestamp": _get(r, "start_time") or 0.0,
                    # FIX-9: visual chunks store human-readable text in display_text
                    # (set by _normalize_result), NOT in the raw "text" field which
                    # often contains JSON blobs or empty strings.
                    "caption": (
                        _get(r, "display_text")
                        or _get(r, "short_caption")
                        or _get(r, "caption")
                        or _get(r, "ocr_text")
                        or _get(r, "board_text")
                        or _get(r, "screen_text")
                        or _get(r, "text")
                        or _get(r, "original_text")
                        or _get(r, "search_text")
                        or ""
                    ),
                    # STEP-0 FIX: carry structured visual fields through to
                    # build_visual_context so counting / spatial / entity
                    # questions see them in the chat (/ask/stream) path.
                    "spatial_layout": _get_visual_field(r, "spatial_layout", ""),
                    "object_counts": _get_visual_field(r, "object_counts", {}),
                    "detected_objects": _get_visual_field(r, "detected_objects", []),
                    "board_text": _get_visual_field(r, "board_text", ""),
                    "screen_text": _get_visual_field(r, "screen_text", ""),
                    "score": _get(r, "score") or 0.0,
                }
                for r in _vis_chunks_in_results
            ],
            key=lambda x: x["timestamp"],
        )
        visual_context = build_visual_context(
            _vis_raw,
            max_items=12,
            max_chars_per_item=300,
        )
        all_ts = _extract_all_timestamps(question)
        if all_ts:
            merged: List[Any] = []
            seen_ids: set = set()
            for ts in all_ts:
                for r in _filter_results_by_timestamp(results, ts, window=60.0):
                    cid = _get(r, "chunk_id") or id(r)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(r)
            results = merged if merged else results
            logger.info(
                "stream timestamp_lookup: %d timestamps, merged to %d chunks",
                len(all_ts), len(results),
            )

    if intent in ("yes_no", "visual") and (max_items is None or max_items < 12):
        max_items = 12

    # STREAM-1: emit metadata preamble immediately
    video_title = (video_metadata or {}).get("title", "") if video_metadata else ""
    yield (
        "event: meta\n"
        "data: " + json.dumps({
            "title": video_title,
            "intent": intent,
            "language": lang,
        }) + "\n\n"
    )

    if not results:
        fb = _extractive_fallback(question, results, intent, lang)
        yield f"data: {json.dumps({'token': fb.answer, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'used_fallback': True, 'citations': [], 'confidence': 0.0})}\n\n"
        return

    configured_model = (
        preferred_model
        or _cfg("ollama_chat_model", None)
        or _cfg("ollama_model", None)
    )
    model_name = _choose_model(configured_model)

    if not model_name:
        fb = _extractive_fallback(question, results, intent, lang)
        yield f"data: {json.dumps({'token': fb.answer, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'used_fallback': True, 'citations': [], 'confidence': 0.0})}\n\n"
        return

    # SPECULATIVE-1: resolve graph_context from future if provided
    _resolved_graph_context = graph_context or ""
    if graph_context_future is not None:
        try:
            _future_result = graph_context_future.result(timeout=5.0)
            if _future_result:
                _resolved_graph_context = _future_result
        except _cf.TimeoutError:
            logger.info("speculative stream: graph_context_future timed out")
        except Exception as _fut_exc:
            logger.warning("speculative stream: graph_context_future raised %s", _fut_exc)

    _max_items = max_items if max_items is not None else 8
    compressed_context, citations = _compress_results_for_prompt(results, max_items=_max_items, question=question)
    confidence = _confidence_score(citations)   # CONF-1

    video_domain = (video_metadata or {}).get("domain", "") if video_metadata else ""
    video_context_block = _build_video_context_block(video_metadata, lang)
    system_prompt = _build_system_prompt(intent, lang, video_title, video_domain, question)
    user_prompt = _build_user_prompt(
        question, compressed_context, intent, lang,
        graph_context=_resolved_graph_context,
        visual_context=visual_context or "",
        video_context_block=video_context_block,
    )

    # BENCHMARK: when a benchmark is specified, override with the dataset-tuned
    # prompt. We reuse the SAME assembled evidence (transcript context + visual
    # context + graph context) so retrieval is unchanged — only the instruction
    # shape differs (Video-MMLU: short/exact/verbatim-formula; EduVidQA: thorough
    # reasoned paragraph).
    if benchmark:
        try:
            from core.benchmark_prompts import get_benchmark_prompt
            _evidence_parts = []
            if visual_context and visual_context.strip():
                _evidence_parts.append("On-screen / visual evidence:\n" + visual_context.strip())
            if _resolved_graph_context and str(_resolved_graph_context).strip():
                _evidence_parts.append("Knowledge-graph evidence:\n" + str(_resolved_graph_context).strip())
            if compressed_context and str(compressed_context).strip():
                _evidence_parts.append("Transcript evidence:\n" + str(compressed_context).strip())
            _evidence = "\n\n".join(_evidence_parts) if _evidence_parts else "(no evidence retrieved)"
            _bsys, _buser = get_benchmark_prompt(benchmark, question, _evidence)
            if _bsys and _buser:
                system_prompt = _bsys
                user_prompt = _buser
        except Exception as _be:
            logger.warning("benchmark prompt override failed (ignored): %s", _be)

    # CONV-3: build history messages
    history_messages = _build_history_messages(conversation_history)

    # Build Ollama messages list with history injected
    messages = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    # summary/translate/list push more tokens (longer system prompt + excerpts + Arabic text)
    _num_ctx = 8192 if intent in ("summary", "translate", "list") else 4096

    base = _get_ollama_base_url()
    url = f"{base}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": _get_temperature(intent),
            "num_predict": _get_num_predict(intent, len(results)),
            "num_ctx": _num_ctx,
            "top_p": DEFAULT_TOP_P,
            "repeat_penalty": DEFAULT_REPEAT_PENALTY,
            "seed": _get_seed(),
        },
    }

    accumulated: List[str] = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=_timeout()) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message") or {}
                token: str = msg.get("content") or ""

                if "<think>" in token or "</think>" in token:
                    token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL)
                    token = re.sub(r"</?think>", "", token)

                if token:
                    accumulated.append(token)
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                if chunk.get("done"):
                    break

        full_answer = "".join(accumulated)
        logger.info("Stream OK | model=%s | intent=%s | chars=%d", model_name, intent, len(full_answer))

        # EMPTY-ANSWER FIX: if model returned nothing, fall back to extractive answer
        if not full_answer.strip():
            logger.warning("Stream returned empty answer — triggering extractive fallback | model=%s | intent=%s", model_name, intent)
            fb = _extractive_fallback(question, results, intent, lang, video_metadata=video_metadata)
            yield f"data: {json.dumps({'token': fb.answer, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'used_fallback': True, 'citations': fb.citations, 'confidence': fb.confidence})}\n\n"
            return

        # CONF-2: append disclaimer on streamed answer if low confidence
        if confidence < LOW_CONF_THRESHOLD:
            disclaimer = {
                "ar": "\n\n⚠️ *ملاحظة: لم أجد مقاطع قوية — قد لا تكون الإجابة دقيقة.*",
                "fr": "\n\n⚠️ *Remarque : passages peu pertinents — réponse peut-être inexacte.*",
                "en": "\n\n⚠️ *Note: low-confidence answer — consider checking the video directly.*",
            }.get(lang, "\n\n⚠️ *Note: low-confidence answer.*")
            yield f"data: {json.dumps({'token': disclaimer, 'done': False})}\n\n"

        yield (
            "data: " + json.dumps({
                "token": "",
                "done": True,
                "model": model_name,
                "intent": intent,
                "language": lang,
                "citations": citations,
                "used_fallback": False,
                "confidence": confidence,   # CONF-1
            }) + "\n\n"
        )

    except requests.Timeout:
        logger.warning("Stream TIMEOUT | model=%s | partial=%d chars",
                       model_name, sum(len(t) for t in accumulated))
        if accumulated:
            yield (
                "data: " + json.dumps({
                    "token": "",
                    "done": True,
                    "error": "timeout_partial",
                    "model": model_name,
                    "citations": citations,
                    "used_fallback": False,
                    "confidence": confidence,
                }) + "\n\n"
            )
        else:
            fb = _extractive_fallback(question, results, intent, lang)
            yield f"data: {json.dumps({'token': fb.answer, 'done': False})}\n\n"
            yield (
                "data: " + json.dumps({
                    "token": "",
                    "done": True,
                    "used_fallback": True,
                    "citations": citations,
                    "confidence": fb.confidence,
                }) + "\n\n"
            )

    except Exception as exc:
        logger.error("Stream failed: %s", exc)
        if accumulated:
            yield (
                "data: " + json.dumps({
                    "token": "",
                    "done": True,
                    "error": str(exc),
                    "model": model_name,
                    "citations": citations,
                    "confidence": confidence,
                }) + "\n\n"
            )
        else:
            fb = _extractive_fallback(question, results, intent, lang)
            yield f"data: {json.dumps({'token': fb.answer, 'done': False})}\n\n"
            yield (
                "data: " + json.dumps({
                    "token": "",
                    "done": True,
                    "used_fallback": True,
                    "citations": citations,
                    "confidence": fb.confidence,
                }) + "\n\n"
            )


# ─────────────────────────────────────────────────────────────────────
# QUALITY VALIDATION
# ─────────────────────────────────────────────────────────────────────

def _validate_answer_quality(
        answer: str,
        citations: List[Dict[str, Any]],
        intent: str,
) -> bool:
    if not answer or len(answer.strip()) < 20:
        return False
    if citations and len(citations) >= 2:
        refusal_phrases = [
            "no relevant information", "i cannot find",
            "i don't have enough", "the evidence does not contain",
            "aucune information pertinente", "لم أجد",
        ]
        if any(p in answer.lower() for p in refusal_phrases):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────

def _postprocess_answer(
        answer: str,
        citations: List[Dict[str, Any]],
        intent: str,
        language_hint: str,
        confidence: float = 1.0,       # CONF-2
) -> str:
    text = answer.strip()
    if citations and intent in ("temporal", "timestamp_lookup"):
        primary_ts = citations[0].get("start_ts")
        if primary_ts:
            text = _inject_primary_timestamp(text, primary_ts, language_hint)
    if re.search(r"\d{1,2}:\d{2}", text):
        pass  # timestamps already present — skip footer
    elif citations:
        header = {
            "ar": "\n\n📍 المراجع الزمنية:",
            "fr": "\n\n📍 Références temporelles :",
            "en": "\n\n📍 Timestamps:",
        }.get(language_hint, "\n\n📍 Timestamps:")
        tail = [header]
        for c in citations[:4]:
            txt = (c.get("text") or "").strip()
            if len(txt) > 200:
                txt = txt[:197] + "..."
            tail.append(f"• [{c['start_ts']} → {c['end_ts']}] {txt}")
            for kp in (c.get("key_points") or [])[:2]:
                tail.append(f"  ‣ {kp}")
        text = text + "\n".join(tail)

    # CONF-2: Append natural-language uncertainty disclaimer when confidence is low
    if confidence < LOW_CONF_THRESHOLD:
        disclaimer = {
            "ar": (
                "\n\n⚠️ *ملاحظة: لم أجد مقاطع قوية تغطي هذا السؤال تحديداً — "
                "قد لا تكون الإجابة أعلاه دقيقة. يُنصح بمراجعة الفيديو مباشرةً.*"
            ),
            "fr": (
                "\n\n⚠️ *Remarque : je n'ai pas trouvé de passages suffisamment pertinents "
                "pour cette question — la réponse ci-dessus est peut-être inexacte. "
                "Vérifiez directement dans la vidéo si besoin.*"
            ),
            "en": (
                "\n\n⚠️ *Note: I couldn't find strong matching segments for this question — "
                "the answer above may not be accurate. Consider checking the video directly.*"
            ),
        }.get(language_hint, (
            "\n\n⚠️ *Note: I couldn't find strong matching segments for this question — "
            "the answer above may not be accurate.*"
        ))
        text = text + disclaimer

    return text


# ─────────────────────────────────────────────────────────────────────
# EXTRACTIVE FALLBACK
# ─────────────────────────────────────────────────────────────────────

def _extractive_fallback(
        question: str,
        results: List[Any],
        intent: str,
        language_hint: str,
        video_metadata: Optional[Dict[str, Any]] = None,
) -> LLMAnswer:
    _, citations = _compress_results_for_prompt(results, max_items=8, question=question)
    confidence = _confidence_score(citations)   # CONF-3

    # For summary intent: if we have a stored video summary, return it directly
    if intent == "summary" and video_metadata:
        stored_summary = (video_metadata.get("summary") or "").strip()
        title = (video_metadata.get("title") or "").strip()
        topics = video_metadata.get("topics") or []
        if stored_summary:
            if language_hint == "ar":
                header = f"**{title}**\n\n" if title else ""
                topics_line = f"\n\n**المواضيع:** {', '.join(topics[:6])}" if topics else ""
                answer = header + stored_summary + topics_line
            elif language_hint == "fr":
                header = f"**{title}**\n\n" if title else ""
                topics_line = f"\n\n**Thèmes :** {', '.join(topics[:6])}" if topics else ""
                answer = header + stored_summary + topics_line
            else:
                header = f"**{title}**\n\n" if title else ""
                topics_line = f"\n\n**Topics:** {', '.join(topics[:6])}" if topics else ""
                answer = header + stored_summary + topics_line
            return LLMAnswer(
                answer=answer, model=None, used_fallback=True,
                citations=citations, intent=intent, language_hint=language_hint,
                raw=None, confidence=max(confidence, 0.5),
            )

    if not citations:
        no_result_msg = {
            "ar": "لم أجد أي مقاطع ذات صلة بسؤالك في هذا الفيديو.",
            "fr": "Je n'ai trouvé aucun passage pertinent à votre question dans cette vidéo.",
            "en": "I couldn't find any relevant passages for your question in this video.",
        }.get(language_hint, "I couldn't find any relevant passages for your question in this video.")
        return LLMAnswer(
            answer=no_result_msg, model=None, used_fallback=True,
            citations=[], intent=intent, language_hint=language_hint,
            raw=None, confidence=0.0,
        )
    lines = []
    headers = {
        ("temporal", "ar"): "⏱️ المقاطع الأكثر صلة بالتوقيت:",
        ("temporal", "fr"): "⏱️ Passages les plus pertinents (chronologie) :",
        ("temporal", "en"): "⏱️ Most relevant moments (chronological):",
        ("timestamp_lookup", "ar"): "🕐 المحتوى عند هذا التوقيت:",
        ("timestamp_lookup", "fr"): "🕐 Contenu à ce moment :",
        ("timestamp_lookup", "en"): "🕐 Content at that timestamp:",
        ("list", "ar"): "📋 العناصر المستخرجة من الفيديو:",
        ("list", "fr"): "📋 Éléments trouvés dans la vidéo :",
        ("list", "en"): "📋 Items found in the video:",
        ("yes_no", "ar"): "🔍 الأدلة المتوفرة:",
        ("yes_no", "fr"): "🔍 Preuves disponibles :",
        ("yes_no", "en"): "🔍 Available evidence:",
        ("factual", "ar"): "🎯 أقوى الأدلة من الفيديو:",
        ("factual", "fr"): "🎯 Preuves les plus pertinentes :",
        ("factual", "en"): "🎯 Strongest evidence from the video:",
        ("summary", "ar"): "📝 ملخص المحتوى:",
        ("summary", "fr"): "📝 Résumé du contenu :",
        ("summary", "en"): "📝 Content summary:",
        ("translate", "ar"): "🌐 ترجمة المقاطع:",
        ("translate", "fr"): "🌐 Traduction des passages :",
        ("translate", "en"): "🌐 Translated passages:",
    }
    header = headers.get(
        (intent, language_hint),
        headers.get(("factual", language_hint), "🎯 Strongest evidence:")
    )
    lines.append(header)
    lines.append("")
    # FIX-VISUAL-3a: Normalize scores relative to the top citation so display
    # reads "100%" for the best match instead of "1%" (RRF scores are tiny by design).
    _top_score = max((c.get("score", 0) for c in citations[:8]), default=1.0) or 1.0
    for i, c in enumerate(citations[:8], start=1):
        txt = (c.get("text") or "").strip()
        if len(txt) > 300:
            txt = txt[:297] + "..."
        speaker = c.get("speaker") or ""
        speaker_prefix = ""
        if speaker and speaker not in ("unknown", "SPEAKER_00", ""):
            speaker_prefix = f" ({speaker})"
        # FIX-VISUAL-3b: Add visual frame indicator for visual chunks
        chunk_type = c.get("chunk_type") or "speech"
        type_badge = " 🖼" if chunk_type == "visual" else ""
        score_pct = f"{c.get('score', 0) / _top_score:.0%}"
        lines.append(
            f"**{i}.** [{c['start_ts']} → {c['end_ts']}]{speaker_prefix}{type_badge} "
            f"(relevance: {score_pct})"
        )
        lines.append(f"   {txt}")
        for kp in (c.get("key_points") or [])[:3]:
            lines.append(f"   • {kp}")
        lines.append("")
    return LLMAnswer(
        answer="\n".join(lines), model=None, used_fallback=True,
        citations=citations, intent=intent, language_hint=language_hint,
        raw=None, confidence=confidence,
    )


# ═════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════

def synthesize_answer(
        question: str,
        results: List[Any],
        *,
        preferred_model: Optional[str] = None,
        video_id: Optional[str] = None,
        language_hint: Optional[str] = None,
        graph_context: Optional[str] = None,
        visual_context: Optional[str] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        max_items: Optional[int] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,   # CONV-2
        modality: Optional[str] = None,   # VISUAL-FIX: 'speech'|'visual'|'both'
        **_extra: Any,
) -> LLMAnswer:
    """
    Blocking answer synthesis.
    v7.0: multi-timestamp, conversation history, confidence scoring.

    VISUAL-FIX: accepts `modality` from classify_query_modality so that
    visual questions are framed correctly for the LLM even when
    _detect_intent() returns 'yes_no' or 'factual' (e.g. "is there a flag?").
    Also auto-builds visual_context from visual chunks in `results` when
    the caller does not supply one explicitly.
    """
    results = results or []
    intent = _detect_intent(question)
    lang = language_hint or _detect_language_hint(question, results)

    # VISUAL-FIX: if the modality classifier says visual/both, upgrade the
    # intent for prompt-building so the LLM gets the right system-prompt
    # framing (describe what is SHOWN, not what was SAID).
    # LVBENCH: counting/event/key_info/temporal_grounding keep their own prompts.
    _lv_modality = lvbench_modality(intent) if intent in _LV_VISUAL_INTENTS else None
    _effective_modality = modality or _lv_modality or ("visual" if intent == "visual" else "both")
    _vis_chunks_in_results = [
        r for r in results if (_get(r, "chunk_type") or "speech") == "visual"
    ]
    has_visual_chunks = bool(_vis_chunks_in_results)

    if _effective_modality in ("visual", "both") and intent not in (
        "visual", "timestamp_lookup",
    ) and intent not in _LV_VISUAL_INTENTS and has_visual_chunks:
        # Upgrade intent to "visual" when modality classifier says so AND
        # visual chunks were actually retrieved. This covers:
        #   - yes/no questions about on-screen content ("is there a flag?")
        #   - factual questions about what appears ("what objects are on the table?")
        #   - list questions about visible items ("what slides are shown?")
        #   - temporal questions about visual events ("when does the logo appear?")
        # In all these cases the LLM must look at frame captions, not speech.
        if _effective_modality == "visual" or intent in ("yes_no", "factual", "list", "temporal"):
            intent = "visual"

    # VISUAL-FIX: auto-build visual_context from ALL visual chunks in results,
    # sorted chronologically. Use max_items=12 so long videos have enough
    # frame coverage — build_visual_context truncates per-caption text anyway.
    if not visual_context and _vis_chunks_in_results:
        _vis_raw = sorted(
            [
                {
                    "timestamp": _get(r, "start_time") or 0.0,
                    # FIX-9: prefer display_text (set by _normalize_result) over
                    # raw text field which is often a JSON blob for visual chunks.
                    "caption": (
                        _get(r, "display_text")
                        or _get(r, "short_caption")
                        or _get(r, "caption")
                        or _get(r, "ocr_text")
                        or _get(r, "board_text")
                        or _get(r, "screen_text")
                        or _get(r, "text")
                        or _get(r, "original_text")
                        or _get(r, "search_text")
                        or ""
                    ),
                    # STEP-0 FIX: carry structured visual fields through to
                    # build_visual_context (non-stream synthesize path).
                    "spatial_layout": _get_visual_field(r, "spatial_layout", ""),
                    "object_counts": _get_visual_field(r, "object_counts", {}),
                    "detected_objects": _get_visual_field(r, "detected_objects", []),
                    "board_text": _get_visual_field(r, "board_text", ""),
                    "screen_text": _get_visual_field(r, "screen_text", ""),
                    "score": _get(r, "score") or 0.0,
                }
                for r in _vis_chunks_in_results
            ],
            key=lambda x: x["timestamp"],
        )
        visual_context = build_visual_context(
            _vis_raw,
            max_items=12,
            max_chars_per_item=300,
        )

    # MULTI-TS-2: collect ALL timestamps and merge results from each window
    if intent == "timestamp_lookup":
        all_ts = _extract_all_timestamps(question)
        if all_ts:
            merged: List[Any] = []
            seen_ids: set = set()
            for ts in all_ts:
                for r in _filter_results_by_timestamp(results, ts, window=60.0):
                    cid = _get(r, "chunk_id") or id(r)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(r)
            results = merged if merged else results
            logger.info(
                "timestamp_lookup: %d timestamps requested, merged to %d chunks",
                len(all_ts), len(results),
            )

    if intent in ("yes_no", "visual") and (max_items is None or max_items < 12):
        max_items = 12

    if not results:
        return _extractive_fallback(question, results, intent, lang)

    configured_model = (
        preferred_model
        or _cfg("ollama_chat_model", None)
        or _cfg("ollama_model", None)
    )
    model_name = _choose_model(configured_model)
    if not model_name:
        return _extractive_fallback(question, results, intent, lang)

    _max_items = max_items if max_items is not None else 8

    video_title = (video_metadata or {}).get("title", "") if video_metadata else ""
    video_domain = (video_metadata or {}).get("domain", "") if video_metadata else ""
    video_context_block = _build_video_context_block(video_metadata, lang)

    compressed_context, citations = _compress_results_for_prompt(results, max_items=_max_items, question=question)
    confidence = _confidence_score(citations)   # CONF-1

    system_prompt = _build_system_prompt(intent, lang, video_title, video_domain, question)
    user_prompt = _build_user_prompt(
        question, compressed_context, intent, lang,
        graph_context=graph_context or "",
        visual_context=visual_context or "",
        video_context_block=video_context_block,
    )

    # CONV-3: build history messages and inject into the Ollama call
    history_messages = _build_history_messages(conversation_history)

    answer_text, raw = _call_ollama_chat(
        model=model_name, system_prompt=system_prompt, user_prompt=user_prompt,
        intent=intent, num_results=len(results),
        history_messages=history_messages,
    )
    if answer_text:
        final = _postprocess_answer(answer_text, citations, intent, lang, confidence=confidence)
        return LLMAnswer(
            answer=final, model=model_name, used_fallback=False,
            citations=citations, intent=intent, language_hint=lang,
            raw=raw, confidence=confidence,
        )
    return _extractive_fallback(question, results, intent, lang)


def answer_question(
        question: str,
        search_results: List[Any],
        preferred_model: Optional[str] = None,
        video_id: Optional[str] = None,
        language_hint: Optional[str] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,   # CONV-2
        graph_context: Optional[str] = None,
        visual_context: Optional[str] = None,
        modality: Optional[str] = None,   # VISUAL-FIX: forward to synthesize_answer
) -> Dict[str, Any]:
    """Backward-compatible wrapper returning a dict."""
    out = synthesize_answer(
        question=question, results=search_results,
        preferred_model=preferred_model, video_id=video_id,
        language_hint=language_hint,
        graph_context=graph_context,
        visual_context=visual_context,
        video_metadata=video_metadata,
        conversation_history=conversation_history,
        modality=modality,
    )
    return {
        "answer": out.answer,
        "model": out.model,
        "used_fallback": out.used_fallback,
        "intent": out.intent,
        "language_hint": out.language_hint,
        "citations": out.citations,
        "confidence": out.confidence,   # CONF-1
    }


# ═════════════════════════════════════════════════════════════════════
# VLM-3: Build visual_context strings
# ═════════════════════════════════════════════════════════════════════

def build_visual_context(
        vision_results: List[Dict[str, Any]],
        *,
        max_items: int = 5,
        max_chars_per_item: int = 240,
) -> str:
    if not vision_results:
        return ""
    def mmss(t: float) -> str:
        t = max(0.0, float(t))
        m, s = divmod(int(round(t)), 60)
        return f"{m:02d}:{s:02d}"
    lines = []
    for i, r in enumerate(vision_results[:max_items], 1):
        ts = mmss(r.get("timestamp", 0.0))
        cap = (r.get("caption") or r.get("short_caption") or r.get("ocr_text") or "").strip()
        if len(cap) > max_chars_per_item:
            cap = cap[:max_chars_per_item - 1] + "…"
        tags = r.get("tags") or []
        tag_s = (" #" + " #".join(tags[:5])) if tags else ""
        score = r.get("score")
        score_s = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""

        # STEP-0 FIX: surface the structured visual fields that the VLM already
        # extracted at index time (spatial_layout, object_counts, board/screen
        # text). These were previously dropped here, so the chat path answered
        # counting / spatial / entity questions from caption text alone.
        extras: List[str] = []
        spatial = (r.get("spatial_layout") or "").strip()
        if spatial:
            if len(spatial) > max_chars_per_item:
                spatial = spatial[:max_chars_per_item - 1] + "…"
            extras.append(f"layout: {spatial}")
        counts = r.get("object_counts") or {}
        if isinstance(counts, dict) and counts:
            count_s = ", ".join(f"{k}: {v}" for k, v in counts.items())
            extras.append(f"counts: {count_s}")
        objs = r.get("detected_objects") or r.get("objects") or []
        if isinstance(objs, (list, tuple)) and objs:
            obj_s = ", ".join(str(o) for o in objs[:12] if str(o).strip())
            if obj_s:
                extras.append(f"objects: {obj_s}")
        # VERBATIM FIX: board_text holds the EXACT on-screen content (formulas,
        # code, algorithms, multi-line bullet lists). Truncating it at
        # max_chars_per_item (e.g. 300) cut off later formulas — e.g. on a slide
        # listing sigmoid+tanh+ReLU (338 chars) the answer kept only sigmoid and
        # reported the others as "not mentioned". Do NOT truncate board_text; give
        # it a large independent budget so multi-formula / code slides survive.
        VERBATIM_CAP = 2000
        board = (r.get("board_text") or "").strip()
        if board and board.upper() not in ("BLANK", "NONE"):
            if len(board) > VERBATIM_CAP:
                board = board[:VERBATIM_CAP - 1] + "…"
            extras.append(f"on-screen text: {board}")
        screen = (r.get("screen_text") or "").strip()
        if screen and screen != board and screen.upper() not in ("BLANK", "NONE"):
            if len(screen) > VERBATIM_CAP:
                screen = screen[:VERBATIM_CAP - 1] + "…"
            extras.append(f"other on-screen text: {screen}")

        extra_s = (" | " + " | ".join(extras)) if extras else ""
        lines.append(f"[{i}] [{ts}]{score_s} {cap}{tag_s}{extra_s}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Lightweight chat helpers for GraphRAG
# ═══════════════════════════════════════════════════════════════════════

def chat_plain(
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        num_predict: int = 256,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
) -> str:
    """Minimal one-shot text completion via Ollama /api/chat."""
    try:
        import requests as _rq
    except Exception:
        return ""
    chosen = model or _choose_model(_cfg("ollama_chat_model", None))
    if not chosen:
        return ""
    base = _get_ollama_base_url().rstrip("/")
    payload = {
        "model": chosen,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "stream": False,
        "think": False,
        "options": {"temperature": float(temperature), "num_predict": int(num_predict), "top_p": 0.9, "seed": _get_seed()},
    }
    try:
        resp = _rq.post(f"{base}/api/chat", json=payload, timeout=int(timeout or _timeout()))
        if resp.status_code != 200:
            return ""
        data = resp.json() or {}
        text = (
            (data.get("message", {}) or {}).get("content")
            or data.get("response") or ""
        )
        return _strip_thinking_tags(text or "").strip()
    except Exception as exc:
        logger.warning("chat_plain failed: %s", exc)
        return ""


def chat_json(
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        num_predict: int = 512,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Strict-JSON completion via Ollama /api/generate (format='json')."""
    try:
        import requests as _rq
        import json as _json
    except Exception:
        return None
    chosen = model or _choose_model(_cfg("ollama_chat_model", None))
    if not chosen:
        return None
    base = _get_ollama_base_url().rstrip("/")
    full_prompt = (system + "\n\n" + prompt) if system else prompt
    payload = {
        "model": chosen,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": float(temperature), "num_predict": int(num_predict), "top_p": 0.9, "seed": _get_seed()},
    }
    try:
        resp = _rq.post(f"{base}/api/generate", json=payload, timeout=int(timeout or _timeout()))
        if resp.status_code != 200:
            return None
        raw = (resp.json() or {}).get("response", "").strip()
        if not raw:
            return None
        try:
            return _json.loads(raw)
        except Exception:
            import re as _re
            m = _re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    return _json.loads(m.group(0))
                except Exception:
                    return None
            return None
    except Exception as exc:
        logger.warning("chat_json failed: %s", exc)
        return None
        return None