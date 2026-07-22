"""
core/search.py — AgenticSearch pipeline v3.1
==============================================

BUG-3 FIX (v3.0 → v3.1):
    AgenticSearch._generate_answer() previously re-implemented the full
    Ollama prompt/call/post-process pipeline that already exists in
    llm_client.py.  The duplication caused:

      • different answer formats depending on which code path was taken
      • different citation styles
      • different quality (system prompt, token budget, think=False, etc.)
      • a maintenance disaster — any fix to llm_client had to be mirrored
        manually in search.py

    FIX: _generate_answer() now delegates 100% to
    llm_client.synthesize_answer() (preferred) or
    llm_client.answer_question() (backward-compat wrapper).

    The duplicated _try_ollama_direct() helper is REMOVED.
    _extractive_fallback() is kept as a last-resort (when llm_client
    is not injected AND ollama is not available), but it no longer
    duplicates the Ollama call logic — it just surfaces the raw segments.

Previous fixes (v3.0) preserved:
    BUG-1  Injection of store/embedder/reranker/llm_client dependencies.
    BUG-3  _build_context() uses _get() safe helper + dynamic start_fmt.
    BUG-4  answer() accepts pre-searched results from api/main.py.
    BUG-5  Aliases answer/ask/synthesize/run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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


# ──────────────────────────────────────────────────────────────────────
# Safe imports
# ──────────────────────────────────────────────────────────────────────

try:
    from core.storage import SearchResult
except ImportError:
    try:
        from core.contracts import SearchResult
    except ImportError:
        @dataclass
        class SearchResult:
            chunk_id: str = ""
            video_id: str = ""
            text: str = ""
            score: float = 0.0
            start_time: Optional[float] = None
            end_time: Optional[float] = None
            language: Optional[str] = None
            speaker: Optional[str] = None
            chunk_type: str = "speech"
            strategy: str = "hybrid"
            dense_score: float = 0.0
            sparse_score: float = 0.0
            key_points: List[str] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Safe getattr that works on dict, DotDict, and dataclass/object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_ts(seconds: Any) -> str:
    try:
        s = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    total = int(round(s))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _detect_language(text: str) -> str:
    """Quick heuristic language detection."""
    if not text:
        return "en"
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars > len(text) * 0.2:
        return "ar"
    french_marks = sum(1 for c in text if c in "àâäéèêëïîôùûüçœæ")
    if french_marks >= 2:
        return "fr"
    text_lower = f" {text.lower()} "
    fr_phrases = (" c'est ", " n'est ", " est-ce ", " qu'il ", " d'un ")
    if any(p in text_lower for p in fr_phrases):
        return "fr"
    return "en"


# ══════════════════════════════════════════════════════════════════════
# AgenticResponse
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgenticResponse:
    question: str = ""
    answer: str = ""
    language: str = "en"
    confidence_score: float = 0.0
    relevant_segments: List[Any] = field(default_factory=list)
    processing_time: float = 0.0
    model: Optional[str] = None
    used_fallback: bool = False
    intent: str = "factual"
    language_hint: str = "en"
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# AgenticSearch
# ══════════════════════════════════════════════════════════════════════

class AgenticSearch:
    """
    RAG pipeline v3.1 for video Q&A.

    BUG-3 FIX: _generate_answer() delegates entirely to
    llm_client.synthesize_answer() / llm_client.answer_question().
    The duplicated Ollama call (_try_ollama_direct) is removed.

    BUG-1 FIX: Accepts injected dependencies (store, embedder, reranker,
    llm_client).
    BUG-4 FIX: answer() accepts pre-searched results from api/main.py.
    """

    def __init__(
        self,
        store: Any = None,
        embedder: Any = None,
        reranker: Any = None,
        llm_client: Any = None,
        language: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_k: int = 5,
        **_kwargs: Any,
    ):
        """
        BUG-1 FIX: accepts injected dependencies.
        Falls back to creating its own instances only if not provided.
        """
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.llm_client = llm_client

        self.language = language or _cfg("default_language", "en")
        self.llm_model = (
            llm_model
            or _cfg("ollama_chat_model")
            or _cfg("ollama_model")
            or "qwen3.5:2b"
        )
        self.default_top_k = top_k or _cfg("search_top_k", 5) or 5

        # Lazy init: only create if not injected
        if self.store is None:
            self._try_init_store()
        if self.embedder is None:
            self._try_init_embedder()

        logger.info(
            "✅ AgenticSearch v3.1 | llm_model=%s | lang=%s | "
            "injected=[store=%s, embedder=%s, reranker=%s, llm=%s]",
            self.llm_model, self.language,
            self.store is not None,
            self.embedder is not None,
            self.reranker is not None,
            self.llm_client is not None,
        )

    def _try_init_store(self) -> None:
        try:
            from core.storage import VideoStore
            self.store = VideoStore()
        except Exception as exc:
            logger.warning("AgenticSearch: could not create VideoStore: %s", exc)

    def _try_init_embedder(self) -> None:
        try:
            from core.embeddings import EmbeddingGenerator
            self.embedder = EmbeddingGenerator()
        except Exception as exc:
            logger.warning("AgenticSearch: could not create EmbeddingGenerator: %s", exc)

    # ==================================================================
    # Public API — multiple entry points for api/main.py compatibility
    # ==================================================================

    def answer(
        self,
        question: str = "",
        results: Optional[List[Any]] = None,
        video_id: Optional[str] = None,
        language_hint: Optional[str] = None,
        top_k: Optional[int] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Main entry point.
        BUG-4 FIX: if results are provided (pre-searched by api/main.py),
        skip the embed+search step and go straight to LLM synthesis.
        """
        t0 = time.time()
        lang = language_hint or _detect_language(question) or self.language

        # Step 1: get search results
        if results and len(results) > 0:
            segments = results
        else:
            segments = self._do_search(question, video_id, top_k)

        if not segments:
            return self._empty_response(question, lang, time.time() - t0)

        # Step 2: rerank if reranker available
        if self.reranker is not None and segments:
            try:
                rerank_fn = getattr(self.reranker, "rerank", None)
                if callable(rerank_fn):
                    segments = rerank_fn(question, segments, top_k=top_k or self.default_top_k)
            except Exception as exc:
                logger.warning("AgenticSearch rerank failed: %s", exc)

        # Step 3: generate answer via llm_client (BUG-3 FIX)
        answer_text, confidence, model_used, citations = self._generate_answer(
            question, segments, lang,
        )

        elapsed = time.time() - t0

        return {
            "answer": answer_text,
            "model": model_used,
            "used_fallback": model_used is None,
            "intent": self._detect_intent(question),
            "language_hint": lang,
            "citations": citations,
            "confidence": confidence,
            "processing_time": round(elapsed, 2),
        }

    # Aliases for api/main.py _call() compatibility
    def ask(self, **kw) -> Dict[str, Any]:
        return self.answer(**kw)

    def synthesize(self, **kw) -> Dict[str, Any]:
        return self.answer(**kw)

    def run(self, **kw) -> Dict[str, Any]:
        return self.answer(**kw)

    def search(
        self,
        question: str = "",
        video_id: Optional[str] = None,
        top_k: Optional[int] = None,
        **kw,
    ) -> Any:
        """
        Full search+answer pipeline (legacy API).
        Returns AgenticResponse for backward compatibility.
        """
        result = self.answer(
            question=question, video_id=video_id, top_k=top_k, **kw,
        )

        return AgenticResponse(
            question=question,
            answer=result.get("answer", ""),
            language=result.get("language_hint", self.language),
            confidence_score=result.get("confidence", 0.0),
            relevant_segments=[],
            processing_time=result.get("processing_time", 0.0),
            model=result.get("model"),
            used_fallback=result.get("used_fallback", False),
            intent=result.get("intent", "factual"),
            language_hint=result.get("language_hint", "en"),
            citations=result.get("citations", []),
            metadata={"video_id": video_id},
        )

    # ==================================================================
    # Internal: search
    # ==================================================================

    def _do_search(
        self, question: str, video_id: Optional[str], top_k: Optional[int],
    ) -> List[Any]:
        """Search using injected store + embedder.

        For translation requests we strip the meta-prefix (e.g. "translate
        the summary of this video in arabic") down to a topic query so the
        embedder searches for *content* rather than the translation directive.
        """
        if self.embedder is None or self.store is None:
            logger.warning("AgenticSearch: embedder or store not available")
            return []

        # Strip translation / language meta-prefix to improve retrieval
        search_query = question
        import re as _re
        _translate_prefix_re = _re.compile(
            r"^\s*(?:can\s+you\s+|please\s+)?(?:translat\w*|traduction|ترجم(?:ة)?)\s+"
            r"(?:the\s+|la\s+|le\s+|)?(?:summary\s+of\s+|résumé\s+de\s+|ملخص\s+)?",
            _re.IGNORECASE,
        )
        _lang_suffix_re = _re.compile(
            r"\s+(?:in(?:to)?\s+)?(?:arabic|french|english|arabe|français|anglais"
            r"|العربية|الفرنسية|الإنجليزية)\s*\??\s*$",
            _re.IGNORECASE,
        )
        stripped = _translate_prefix_re.sub("", question)
        stripped = _lang_suffix_re.sub("", stripped).strip(" ?!")
        if stripped and len(stripped) >= 3:
            search_query = stripped

        try:
            query_emb = self.embedder.embed_query(search_query)
            results = self.store.search(
                query_embedding=query_emb,
                query=search_query,
                video_id=video_id,
                top_k=top_k or self.default_top_k,
            )
            return results or []
        except Exception as exc:
            logger.error("AgenticSearch search failed: %s", exc)
            return []

    # ==================================================================
    # Internal: context building
    # ==================================================================

    def _build_context(self, segments: List[Any], lang: str = "en") -> str:
        """
        Safe context builder using _get() helper (works on dict, DotDict,
        dataclass). Computes start_fmt dynamically if not present.

        BUG-5 FIX: Sort segments chronologically before building context
        so the LLM always reads them in video order.
        """
        if not segments:
            return "No relevant segments found."

        def _seg_start(s) -> float:
            t = _get(s, "start_time") or _get(s, "start") or 0
            try:
                return float(t)
            except (TypeError, ValueError):
                return 0.0

        segments = sorted(segments, key=_seg_start)

        lines = []
        for seg in segments:
            ts = (
                _get(seg, "start_fmt")
                or _get(seg, "timestamp_formatted")
                or _fmt_ts(_get(seg, "start_time", 0))
            )
            text = (
                _get(seg, "text")
                or _get(seg, "original_text")
                or _get(seg, "search_text")
                or ""
            )
            speaker = _get(seg, "speaker", "")
            language = _get(seg, "language", "")

            kp = _get(seg, "key_points", [])
            kp_text = ""
            if kp and isinstance(kp, list) and len(kp) > 0:
                kp_items = [f"  • {k}" for k in kp[:3] if k]
                if kp_items:
                    kp_text = "\n" + "\n".join(kp_items)

            prefix = f"[{ts}]"
            if speaker and speaker != "unknown" and speaker != "SPEAKER_00":
                prefix = f"[{ts} | {speaker}]"
            if language and language not in ("unknown", "en", lang):
                prefix += f" ({language})"

            lines.append(f"{prefix} {text.strip()}{kp_text}")

        return "\n\n".join(lines)

    # ==================================================================
    # Internal: answer generation — BUG-3 FIX
    # ==================================================================

    def _generate_answer(
        self, question: str, segments: List[Any], lang: str,
    ) -> Tuple[str, float, Optional[str], List[Dict[str, Any]]]:
        """
        BUG-3 FIX: Delegate entirely to llm_client.synthesize_answer() or
        llm_client.answer_question().  No Ollama call is made here.

        Eliminates the duplicated prompt-building, Ollama call, and
        post-processing that previously lived in _try_ollama_direct().

        Returns (answer_text, confidence, model_name_or_None, citations).
        """

        # ── Priority 1: llm_client.synthesize_answer() ────────────────
        # This is the canonical, fully-featured path: system prompt,
        # intent detection, quality validation, timestamp injection,
        # think=False, model selection — all handled by llm_client.
        if self.llm_client is not None:
            synth_fn = getattr(self.llm_client, "synthesize_answer", None)
            if callable(synth_fn):
                try:
                    result = synth_fn(
                        question=question,
                        results=segments,
                        language_hint=lang,
                        preferred_model=self.llm_model,
                    )
                    # synthesize_answer returns an LLMAnswer dataclass
                    answer = getattr(result, "answer", None)
                    if answer:
                        return (
                            answer,
                            0.0 if getattr(result, "used_fallback", False) else 0.85,
                            getattr(result, "model", None),
                            getattr(result, "citations", []),
                        )
                except Exception as exc:
                    logger.warning(
                        "llm_client.synthesize_answer failed: %s — trying answer_question",
                        exc,
                    )

            # ── Priority 2: llm_client.answer_question() (compat wrapper) ──
            answer_fn = getattr(self.llm_client, "answer_question", None)
            if callable(answer_fn):
                try:
                    result = answer_fn(
                        question=question,
                        search_results=segments,
                        preferred_model=self.llm_model,
                        language_hint=lang,
                    )
                    if isinstance(result, dict) and result.get("answer"):
                        return (
                            result["answer"],
                            0.0 if result.get("used_fallback") else 0.85,
                            result.get("model"),
                            result.get("citations", []),
                        )
                except Exception as exc:
                    logger.warning("llm_client.answer_question failed: %s", exc)

        # ── Priority 3: module-level synthesize_answer from llm_client.py ──
        # When llm_client was not injected as an instance we try importing
        # the module-level function directly.  This avoids any Ollama
        # duplication because we're calling the *same* function.
        try:
            from core import llm_client as _lc
            result = _lc.synthesize_answer(
                question=question,
                results=segments,
                language_hint=lang,
                preferred_model=self.llm_model,
            )
            answer = getattr(result, "answer", None)
            if answer:
                return (
                    answer,
                    0.0 if getattr(result, "used_fallback", False) else 0.85,
                    getattr(result, "model", None),
                    getattr(result, "citations", []),
                )
        except Exception as exc:
            logger.warning("module-level llm_client.synthesize_answer failed: %s", exc)

        # ── Priority 4: extractive fallback (no Ollama, no duplication) ──
        return self._extractive_fallback(segments, lang), 0.4, None, self._build_citations(segments)

    # ==================================================================
    # Internal: extractive fallback (last resort — no LLM call)
    # ==================================================================

    def _extractive_fallback(self, segments: List[Any], lang: str) -> str:
        """Last resort: surface top segments verbatim. No LLM, no Ollama."""
        if not segments:
            if lang == "ar":
                return "لم أجد معلومات ذات صلة في الفيديو."
            elif lang == "fr":
                return "Je n'ai pas trouvé d'information pertinente dans la vidéo."
            return "No relevant information found in the video."

        lines = []
        if lang == "ar":
            lines.append("الأدلة المتاحة:")
        elif lang == "fr":
            lines.append("Preuves disponibles:")
        else:
            lines.append("Available evidence:")

        for seg in segments[:4]:
            ts = _get(seg, "start_fmt") or _fmt_ts(_get(seg, "start_time", 0))
            text = (_get(seg, "text") or "").strip()
            if len(text) > 250:
                text = text[:247] + "..."
            lines.append(f"- [{ts}] {text}")

            kp = _get(seg, "key_points", [])
            if kp and isinstance(kp, list):
                for k in kp[:2]:
                    lines.append(f"  • {k}")

        return "\n".join(lines)

    # ==================================================================
    # Internal: citations + intent
    # ==================================================================

    def _build_citations(self, segments: List[Any]) -> List[Dict[str, Any]]:
        citations = []
        for seg in segments[:6]:
            start = _safe_float(_get(seg, "start_time"))
            end = _safe_float(_get(seg, "end_time"))
            citations.append({
                "chunk_id":  _get(seg, "chunk_id"),
                "video_id":  _get(seg, "video_id"),
                "text":      (_get(seg, "text") or "")[:300],
                "start_time": start,
                "end_time":   end,
                "start_fmt":  _get(seg, "start_fmt") or _fmt_ts(start),
                "end_fmt":    _get(seg, "end_fmt") or _fmt_ts(end),
                "language":   _get(seg, "language"),
                "speaker":    _get(seg, "speaker"),
                "score":      _safe_float(_get(seg, "score") or _get(seg, "rerank_score")),
                "key_points": _get(seg, "key_points", []),
            })
        return citations

    def _detect_intent(self, question: str) -> str:
        q = question.lower()
        temporal = ("when", "what time", "before", "after", "first", "last",
                    "متى", "قبل", "بعد", "quand", "avant", "après")
        listing = ("list", "all", "which", "what are", "name",
                   "اذكر", "ما هي", "liste", "quels", "quelles")
        if any(w in q for w in temporal):
            return "temporal"
        if any(w in q for w in listing):
            return "list"
        return "factual"

    def _empty_response(
        self, question: str, lang: str, elapsed: float,
    ) -> Dict[str, Any]:
        if lang == "ar":
            msg = "لم أجد معلومات ذات صلة في الفيديو."
        elif lang == "fr":
            msg = "Aucune information pertinente trouvée."
        else:
            msg = "No relevant information found in the video."
        return {
            "answer": msg,
            "model": None,
            "used_fallback": True,
            "intent": self._detect_intent(question),
            "language_hint": lang,
            "citations": [],
            "confidence": 0.0,
            "processing_time": round(elapsed, 2),
        }

    # ==================================================================
    # Display (optional, for CLI usage)
    # ==================================================================

    def display_response(self, response: Any) -> None:
        if isinstance(response, dict):
            answer = response.get("answer", "")
            conf = response.get("confidence", 0.0)
            citations = response.get("citations", [])
        else:
            answer = getattr(response, "answer", "")
            conf = getattr(response, "confidence_score", 0.0)
            citations = getattr(response, "citations", [])

        print(f"\n{'=' * 70}")
        print(f"ANSWER (confidence: {conf:.0%}):")
        print(f"{'=' * 70}\n")
        print(answer)

        if citations:
            print(f"\n{'=' * 70}")
            print("CITATIONS:")
            for i, c in enumerate(citations, 1):
                ts = c.get("start_fmt", "??:??")
                text = (c.get("text") or "")[:120]
                print(f"  {i}. " f"[{ts}] {text}")
