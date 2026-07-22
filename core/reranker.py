"""
core/reranker.py — Cross-encoder reranker v3.0
================================================
Corrections :

 BUG-1  `from typing import Any` était importé EN BAS du fichier (après usage).
        FIX : import correct en haut.

 BUG-2  rerank() acceptait top_n mais api/main.py appelle avec top_k.
        FIX : accepte les DEUX (top_k alias de top_n) + **kwargs.

 BUG-3  _normalize_results() créait SearchResult avec start_fmt/end_fmt
        kwargs, mais SearchResult dans storage.py v3 n'a PAS ces champs.
        FIX : ne pas passer start_fmt/end_fmt au constructeur, les calculer
        dynamiquement dans le résultat reranké.

 BUG-4  Les résultats peuvent être des DotDict (api/main.py v5.1), des
        dicts, ou des SearchResult. _normalize_results() doit tout gérer.
        FIX : helper _get() qui fonctionne sur dict, DotDict, et objets.

 BUG-5  Le reranked SearchResult utilisait result.start_fmt qui n'existe
        pas sur le SearchResult de storage v3.
        FIX : utiliser _get() safe pour tous les champs optionnels.

 BONUS  Safe settings access. rerank_results() alias pour api/main.py _call().

 DOC7-FIX-1  _score_batch() : ajout du cas fallback backend == "none"
             → retourne [0.0] * len(pairs) pour préserver l'ordre original.
"""

from __future__ import annotations

import logging
import math
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
# Safe import : SearchResult
# ──────────────────────────────────────────────────────────────────────

try:
    from core.contracts import SearchResult
except ImportError:
    try:
        from core.storage import SearchResult
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
            topic: Optional[str] = None
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


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip() or default


def _fmt_ts(seconds: Any) -> str:
    try:
        s = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    total = int(round(s))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

RERANKER_PRESETS: Dict[str, str] = {
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "quality": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    score_threshold: Optional[float] = None
    normalize_scores: bool = True
    auto_load: bool = True


# ══════════════════════════════════════════════════════════════════════
# CrossEncoderReranker
# ══════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Multilingual cross-encoder reranker v3.0.

    Default: BAAI/bge-reranker-v2-m3 (multilingual, lightweight).
    Fallback: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1.

    BUG-2 FIX: accepts both top_k and top_n.
    BUG-3 FIX: doesn't require start_fmt/end_fmt on input results.
    BUG-4 FIX: handles DotDict, dict, and SearchResult inputs.
    DOC7-FIX-1: _score_batch() returns [0.0] * n when backend == "none".
    """

    def __init__(self, config: Optional[RerankerConfig] = None, **_: Any):
        self.config = config or RerankerConfig(
            model_name=_cfg("reranker_model", "BAAI/bge-reranker-v2-m3"),
            device=_cfg("reranker_device", "cpu"),
        )
        self._model = None
        self._backend = "none"
        self._loaded = False
        self._load_attempts = 0

        if self.config.auto_load:
            self._load_model()

    # ── Loading ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._loaded or self._load_attempts >= 2:
            return
        self._load_attempts += 1

        if "bge-reranker" in self.config.model_name.lower():
            if self._try_flag_reranker():
                return

        if self._try_cross_encoder():
            return

        if self._backend == "none":
            self._try_flag_reranker()

    def _try_flag_reranker(self) -> bool:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            logger.debug("FlagEmbedding not available")
            return False

        try:
            logger.info("Loading FlagReranker: %s", self.config.model_name)
            t0 = time.perf_counter()
            self._model = FlagReranker(self.config.model_name, use_fp16=True)
            elapsed = (time.perf_counter() - t0) * 1000
            self._backend = "flag_reranker"
            self._loaded = True
            logger.info("✅ FlagReranker loaded in %.0fms", elapsed)
            return True
        except Exception as exc:
            logger.warning("FlagReranker load failed: %s", exc)
            return False

    def _try_cross_encoder(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("sentence-transformers not installed for reranker")
            return False

        try:
            model_name = self.config.model_name
            if "bge-reranker" in model_name.lower():
                model_name = RERANKER_PRESETS["multilingual"]

            logger.info("Loading CrossEncoder: %s", model_name)
            t0 = time.perf_counter()
            self._model = CrossEncoder(
                model_name,
                device=self.config.device,
                max_length=self.config.max_length,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self._backend = "cross_encoder"
            self._loaded = True
            logger.info("✅ CrossEncoder loaded in %.0fms", elapsed)
            return True
        except Exception as exc:
            logger.warning("CrossEncoder load failed: %s", exc)
            return False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self.config.model_name

    # ── Public API ────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        results: List[Any],
        top_n: Optional[int] = None,
        top_k: Optional[int] = None,
        **_kwargs: Any,
    ) -> List[Any]:
        """
        Rerank search results using the cross-encoder.

        BUG-2 FIX: accepts both top_k and top_n (top_k takes priority).
        BUG-4 FIX: accepts DotDict, dict, SearchResult — returns same type.
        """
        if not results:
            return []

        self._ensure_loaded()

        effective_top = top_k if top_k is not None else (top_n or len(results))

        if not self._loaded:
            logger.warning("Reranker not loaded — falling back to embedding score ranking.")
            # Sort by the best available score from vector search so at least
            # ordering and top-k trimming are meaningful (not random retrieval order).
            def _raw_score(r):
                s = _get(r, "score", None)
                return float(s) if s is not None else 0.0
            sorted_results = sorted(results, key=_raw_score, reverse=True)
            return sorted_results[:effective_top]

        # Skip scoring overhead for trivial cases
        if len(results) <= 1:
            return results[:effective_top]

        t0 = time.perf_counter()

        # Extract texts for scoring
        texts = [self._extract_text(r) for r in results]
        raw_scores = self._score_batch(query, texts)

        if self.config.normalize_scores:
            scores = [self._sigmoid(s) for s in raw_scores]
        else:
            scores = list(raw_scores)

        # Build scored pairs, preserving original result type
        scored: List[Tuple[float, Any]] = []
        for score, result in zip(scores, results):
            if self.config.score_threshold is not None and score < self.config.score_threshold:
                continue

            # Inject rerank score into the result (works for dict, DotDict, object)
            enriched = self._inject_rerank_score(result, score)
            scored.append((score, enriched))

        scored.sort(key=lambda x: -x[0])
        final = [r for _, r in scored[:effective_top]]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Reranking: %d → %d results in %.1fms | backend=%s",
            len(results), len(final), elapsed_ms, self._backend,
        )

        return final

    # Aliases for api/main.py _call() compatibility
    def rerank_results(self, query: str, results: List[Any], **kw) -> List[Any]:
        return self.rerank(query, results, **kw)

    # ── Internal ──────────────────────────────────────────────────────

    def _extract_text(self, result: Any) -> str:
        """Extract text from any result type (dict, DotDict, object)."""
        text = (
            _get(result, "text")
            or _get(result, "original_text")
            or _get(result, "search_text")
            or ""
        )
        return _safe_str(text, " ")

    def _inject_rerank_score(self, result: Any, score: float) -> Any:
        """
        Inject rerank_score into the result WITHOUT creating a new SearchResult.
        This preserves the original type (DotDict, dict, etc.) so downstream
        consumers (AgenticSearch, llm_client) can use their expected access pattern.
        """
        rerank_score = round(float(score), 6)

        if isinstance(result, dict):
            # dict or DotDict — mutate in place
            result["rerank_score"] = rerank_score
            result["score"] = rerank_score
            old_strategy = _safe_str(result.get("strategy"), "hybrid")
            if "+rerank" not in old_strategy:
                result["strategy"] = old_strategy + "+rerank"
            # Add start_fmt/end_fmt if missing (for AgenticSearch compatibility)
            if "start_fmt" not in result:
                result["start_fmt"] = _fmt_ts(_get(result, "start_time"))
            if "end_fmt" not in result:
                result["end_fmt"] = _fmt_ts(_get(result, "end_time"))
            return result
        else:
            # dataclass or object — try setattr
            try:
                result.rerank_score = rerank_score
                result.score = rerank_score
                old_strategy = _safe_str(getattr(result, "strategy", "hybrid"))
                if "+rerank" not in old_strategy:
                    result.strategy = old_strategy + "+rerank"
                if not getattr(result, "start_fmt", None):
                    result.start_fmt = _fmt_ts(getattr(result, "start_time", None))
                if not getattr(result, "end_fmt", None):
                    result.end_fmt = _fmt_ts(getattr(result, "end_time", None))
            except (AttributeError, TypeError):
                pass
            return result

    def _score_batch(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs.

        DOC7-FIX-1: Added fallback case for backend == "none".
        Returns [0.0] * len(pairs) when no backend is loaded,
        which preserves the original result order rather than crashing.
        """
        query = (query or "").strip()
        pairs = [(query, (t or "").strip() or " ") for t in texts]
        scores: List[float] = []
        bs = self.config.batch_size

        if self._backend == "flag_reranker":
            for start in range(0, len(pairs), bs):
                batch = pairs[start:start + bs]
                batch_scores = self._model.compute_score(batch, normalize=False)
                if isinstance(batch_scores, (int, float)):
                    scores.append(float(batch_scores))
                else:
                    scores.extend([float(s) for s in batch_scores])

        elif self._backend == "cross_encoder":
            for start in range(0, len(pairs), bs):
                batch = pairs[start:start + bs]
                result = self._model.predict(batch, show_progress_bar=False)
                if hasattr(result, "tolist"):
                    scores.extend(result.tolist())
                else:
                    scores.extend(list(result))

        else:
            # DOC7-FIX-1 : No backend loaded → neutral scores (preserve original order)
            # _ensure_loaded() was already called but the model is unavailable.
            # Returning 0.0 for all keeps the original ranking intact without crashing.
            logger.debug(
                "Reranker backend == 'none', returning neutral scores "
                "(original order preserved)"
            )
            scores = [0.0] * len(pairs)

        return scores

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            return e / (1.0 + e)

    def warmup(self) -> None:
        self._ensure_loaded()
        if self._loaded:
            self._score_batch("warmup", ["warmup passage"])
            logger.info("Reranker warmed up (%s)", self._backend)