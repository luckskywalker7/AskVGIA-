"""
core/storage.py — FAISS + BM25 hybrid storage v3.1
====================================================

v3.1 changes (vs v3.0):
 -----------------------
 PERF-BM25-1  LanguageBM25Index now exposes add_many() for true BULK ADD.
              _dirty is toggled ONCE for the whole batch instead of once
              per document. This turns "N adds during ingestion" into a
              single O(1) flag flip, and the expensive BM25Okapi() rebuild
              happens AT MOST ONCE — lazily on the first search() call.
              add_embedded_chunks() in VideoStore auto-uses the batch path.

 PERF-BM25-2  _rebuild() is skip-short-circuited when the corpus hasn't
              grown since the last successful build (tracked by
              _last_built_size). Previously any `_dirty=True` triggered a
              full rebuild even if no new tokens were added.

 PERF-BM25-3  _rebuild() now logs duration + corpus size so slow BM25
              builds on 10k+ corpora are visible in the logs:
                 "BM25 rebuild lang=en | docs=12450 | 1.8s"

 PERF-BM25-4  prewarm() — explicit public method that ingestion pipelines
              can call ONCE at the end of add_embedded_chunks() to pay
              the rebuild cost up front instead of on the first user query.
              VideoStore.add_embedded_chunks(..., prewarm_bm25=True) wires
              this in.

Corrections v2.0 → v3.0 :

 CRIT-1  FAISS deletion réelle : IndexIDMap2(IndexFlatIP) supporte
         remove_ids(). Pour le mode HNSW → tombstone set + rebuild
         automatique quand le seuil est dépassé.

 CRIT-2  Détection de langue robuste : utilise lingua (LanguageDetectorBuilder)
         si disponible, sinon fallback amélioré avec comptage de caractères
         Unicode au lieu de mots-clés fragiles comme "est".

 CRIT-3  ScoreCache : SHA-256 (pas de hash faible) + stockage de la clé brute
         pour vérification anti-collision.

 CRIT-4  _rrf_fuse() : dense_score / sparse_score = 0.0 par défaut
         (jamais None → pas de crash dans le frontend JS).

 CRIT-5  get_stats() : champ unifié "total_chunks" pour global ET per-video,
         compatible avec api/main.py v5.1.

 BONUS   Accès settings avec defaults safe → pas de crash si settings manque.
         _call-style safe getattr sur les chunks (pas d'AttributeError).

 DOC7-FIX-1  _rrf_fuse() : track dense_set / sparse_set pour détecter
             correctement les sources même quand dense_score == 0.0.

 DOC7-FIX-2  search() : paramètre filter_tags ajouté (schéma v2.0).
             _passes() filtre par filter_tags si fourni.

 DOC7-FIX-3  get_stats() : champ filter_tags dans les stats per-video.

 FIX-WINDOWS-1  Sauvegarde : remplacer '*' par 'star' dans les noms de fichiers
                (caractère invalide sur Windows).

Public API inchangée :
  add_embedded_chunks(chunks)   — sans video_id kwarg
  search(query_embedding, ...)  — top_k / limit alias
  list_videos()                 — retourne List[str]
  get_stats(video_id=None)      — dict unifié
  delete_video(video_id)
  save()
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Safe settings access
# ──────────────────────────────────────────────────────────────────────

try:
    from config.settings import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any = None) -> Any:
    """Get a setting with a safe fallback."""
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


def _cfg_list(name: str, default: list) -> list:
    val = _cfg(name, default)
    if isinstance(val, list):
        return val
    return default


def _get_fusion_weights(lang: str) -> Tuple[float, float]:
    """Get (dense_weight, sparse_weight) for a language."""
    # Try settings.get_fusion_weights() first
    if _settings is not None and hasattr(_settings, "get_fusion_weights"):
        try:
            return _settings.get_fusion_weights(lang)
        except Exception:
            pass

    lang = (lang or "").strip().lower()[:2]
    weights = {
        "ar": (_cfg_float("fusion_ar_dense", 0.35), _cfg_float("fusion_ar_sparse", 0.65)),
        "en": (_cfg_float("fusion_en_dense", 0.65), _cfg_float("fusion_en_sparse", 0.35)),
        "fr": (_cfg_float("fusion_fr_dense", 0.50), _cfg_float("fusion_fr_sparse", 0.50)),
    }
    return weights.get(lang, (
        _cfg_float("fusion_default_dense", 0.50),
        _cfg_float("fusion_default_sparse", 0.50),
    ))


# ──────────────────────────────────────────────────────────────────────
# Import FAISS
# ──────────────────────────────────────────────────────────────────────
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("faiss-cpu not installed! Run: pip install faiss-cpu")

# ──────────────────────────────────────────────────────────────────────
# Import BM25
# ──────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")

# ──────────────────────────────────────────────────────────────────────
# CRIT-2 : Import lingua for robust language detection
# ──────────────────────────────────────────────────────────────────────
_lingua_detector = None
try:
    from lingua import Language, LanguageDetectorBuilder

    _lingua_detector = (
        LanguageDetectorBuilder
        .from_languages(Language.ARABIC, Language.ENGLISH, Language.FRENCH)
        .with_minimum_relative_distance(0.25)
        .build()
    )
    logger.info("lingua LanguageDetector loaded (ar/en/fr)")
except ImportError:
    logger.info("lingua not installed → using Unicode heuristic for lang detection")
except Exception as exc:
    logger.warning("lingua init failed: %s → using heuristic", exc)


# ──────────────────────────────────────────────────────────────────────
# SearchResult contract
# ──────────────────────────────────────────────────────────────────────
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
        topic: Optional[str] = None
        chunk_type: str = "speech"
        strategy: str = "hybrid"
        dense_score: float = 0.0           # CRIT-4 : jamais None
        sparse_score: float = 0.0          # CRIT-4 : jamais None
        key_points: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Tokenizer — language-aware for BM25
# ======================================================================

_AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_AR_TATWEEL = re.compile(r"\u0640+")


def tokenize(text: str, language: str = "en") -> List[str]:
    """Language-aware tokenization for BM25."""
    if not text:
        return []
    text = text.strip().lower()

    if language in ("ar", "fa", "ur"):
        text = _AR_DIACRITICS.sub("", text)
        text = _AR_TATWEEL.sub("", text)
        text = text.replace("\u0623", "\u0627")  # أ → ا
        text = text.replace("\u0625", "\u0627")  # إ → ا
        text = text.replace("\u0622", "\u0627")  # آ → ا
        text = text.replace("\u0629", "\u0647")  # ة → ه
        tokens = re.split(r"[\s\u200c\u200d.,;:!?\-/\\()«»\"']+", text)
    else:
        tokens = re.split(r"[\s.,;:!?\-/\\()«»\"']+", text)

    return [t for t in tokens if len(t) >= 2]


# ======================================================================
# CRIT-2 : Robust language detection
# ======================================================================

def detect_language(text: str) -> str:
    """
    Robust language detection.
    1) lingua (if available) — high accuracy
    2) Unicode character ratio fallback — safe, no single-keyword false matches
    """
    if not text or not text.strip():
        return _cfg("default_language", "en")

    # 1) lingua
    if _lingua_detector is not None:
        try:
            result = _lingua_detector.detect_language_of(text)
            if result is not None:
                lang_map = {
                    Language.ARABIC: "ar",
                    Language.ENGLISH: "en",
                    Language.FRENCH: "fr",
                }
                detected = lang_map.get(result)
                if detected:
                    return detected
        except Exception:
            pass

    # 2) Unicode heuristic (multi-word phrase matching, no single-word traps)
    total = len(text)
    if total == 0:
        return _cfg("default_language", "en")

    arabic_count = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    latin_count = sum(1 for c in text if "a" <= c.lower() <= "z")

    # Strong Arabic signal
    if arabic_count > total * 0.2:
        return "ar"

    # French: check for accented characters + French-specific bigrams
    french_accented = sum(1 for c in text if c in "àâäéèêëïîôùûüçœæ")
    if french_accented >= 2:
        return "fr"

    # French-specific multi-word patterns (much safer than single words)
    text_lower = f" {text.lower()} "
    french_phrases = (
        " c'est ", " n'est ", " l'est ", " qu'est ",
        " est-ce ", " qu'il ", " qu'elle ", " n'a ",
        " l'a ", " d'un ", " d'une ",
        " je suis ", " nous avons ", " vous êtes ",
        " dans le ", " dans la ", " sur le ", " sur la ",
        " pour le ", " pour la ",
    )
    french_hits = sum(1 for p in french_phrases if p in text_lower)
    if french_hits >= 1:
        return "fr"

    # Default
    if latin_count > total * 0.3:
        return "en"

    return _cfg("default_language", "en")


# ======================================================================
# CRIT-3 : ScoreCache — SHA-256 + raw key verification
# ======================================================================

class ScoreCache:
    """LRU cache with TTL and collision-safe hashing."""

    def __init__(self, max_size: int = 500, ttl: float = 1800.0):
        self._cache: OrderedDict[str, Tuple[float, str, List[Any]]] = OrderedDict()
        # value = (timestamp, raw_key, results)
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, video_id: Optional[str]) -> Tuple[str, str]:
        """Returns (hash_key, raw_key)."""
        raw = f"{query}\x00{video_id or '*'}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return h, raw

    def get(self, query: str, video_id: Optional[str]) -> Optional[List[Any]]:
        h, raw = self._make_key(query, video_id)
        with self._lock:
            if h in self._cache:
                ts, stored_raw, data = self._cache[h]
                # CRIT-3 : vérifier que la clé brute correspond (anti-collision)
                if stored_raw != raw:
                    # hash collision → miss
                    self.misses += 1
                    return None
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(h)
                    self.hits += 1
                    return data
                # expired
                del self._cache[h]
            self.misses += 1
            return None

    def put(self, query: str, video_id: Optional[str], results: List[Any]) -> None:
        h, raw = self._make_key(query, video_id)
        with self._lock:
            self._cache[h] = (time.time(), raw, results)
            self._cache.move_to_end(h)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate_video(self, video_id: str) -> None:
        with self._lock:
            to_del = []
            for k, (_, _, results) in self._cache.items():
                for r in results:
                    vid = getattr(r, "video_id", None)
                    if vid is None and isinstance(r, dict):
                        vid = r.get("video_id")
                    if vid == video_id:
                        to_del.append(k)
                        break
            for k in to_del:
                del self._cache[k]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ======================================================================
# CRIT-1 : LanguageFAISSIndex — real deletion with IndexIDMap2
# ======================================================================

class LanguageFAISSIndex:
    """
    One FAISS index per language.

    Mode "flat" (default):
      IndexIDMap2(IndexFlatIP) → supporte remove_ids() nativement.

    Mode "hnsw":
      IndexHNSWFlat (pas de remove_ids) → tombstone set + rebuild
      automatique quand len(tombstones) > rebuild_threshold.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        rebuild_threshold: int = 100,
    ):
        self.dim = dim
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._ef_construction = ef_construction
        self._rebuild_threshold = rebuild_threshold

        # chunk_id ↔ int64 ID mapping
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._next_id: int = 0

        # Tombstone set for HNSW mode
        self._tombstones: Set[int] = set()

        # Raw vectors for HNSW rebuild
        self._vectors: Dict[int, np.ndarray] = {}   # only used in HNSW mode

        self.index = self._create_index()

    def _create_index(self):
        if not FAISS_AVAILABLE:
            return None

        if self._index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, self._hnsw_m)
            idx.hnsw.efConstruction = self._ef_construction
            # HNSW : on wrappe dans IDMap2 pour l'API uniforme
            # MAIS remove_ids ne fonctionne pas sur HNSW → tombstones
            return faiss.IndexIDMap2(idx)
        else:
            # flat : IndexIDMap2(IndexFlatIP) → remove_ids() fonctionne
            return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _alloc_id(self, chunk_id: str) -> int:
        if chunk_id in self._str_to_int:
            return self._str_to_int[chunk_id]
        int_id = self._next_id
        self._next_id += 1
        self._str_to_int[chunk_id] = int_id
        self._int_to_str[int_id] = chunk_id
        return int_id

    @property
    def size(self) -> int:
        """Nombre de vecteurs actifs (excluant les tombstones)."""
        return len(self._str_to_int) - len(self._tombstones)

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        if self.index is None:
            return
        if chunk_id in self._str_to_int:
            # Déjà présent — si c'était un tombstone, on le ressuscite
            int_id = self._str_to_int[chunk_id]
            if int_id in self._tombstones:
                self._tombstones.discard(int_id)
                # Re-add le vecteur
                vec = vector.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vec)
                ids = np.array([int_id], dtype=np.int64)
                self.index.add_with_ids(vec, ids)
                if self._index_type == "hnsw":
                    self._vectors[int_id] = vec.copy()
            return  # already active

        int_id = self._alloc_id(chunk_id)
        vec = vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        ids = np.array([int_id], dtype=np.int64)
        self.index.add_with_ids(vec, ids)

        if self._index_type == "hnsw":
            self._vectors[int_id] = vec.copy()

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if self.index is None or self.size == 0:
            return []

        vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        # Search more than needed to account for tombstones
        search_k = min(top_k + len(self._tombstones) + 10, self.index.ntotal)
        if search_k <= 0:
            return []

        scores, indices = self.index.search(vec, search_k)
        results = []
        for score, int_id in zip(scores[0], indices[0]):
            if int_id < 0:
                continue
            int_id = int(int_id)
            # Exclure les tombstones
            if int_id in self._tombstones:
                continue
            chunk_id = self._int_to_str.get(int_id)
            if chunk_id is None:
                continue
            results.append((chunk_id, float(score)))
            if len(results) >= top_k:
                break

        return results

    def remove(self, chunk_id: str) -> None:
        """
        CRIT-1 : vraie suppression.
        flat mode  → remove_ids() natif FAISS
        hnsw mode  → tombstone + rebuild si seuil dépassé
        """
        if chunk_id not in self._str_to_int:
            return

        int_id = self._str_to_int[chunk_id]

        if self._index_type == "flat":
            # IndexIDMap2(IndexFlatIP) supporte remove_ids
            try:
                if self.index is not None:
                    self.index.remove_ids(np.array([int_id], dtype=np.int64))
            except Exception as e:
                logger.warning("FAISS remove_ids failed: %s", e)
            del self._str_to_int[chunk_id]
            del self._int_to_str[int_id]

        else:
            # HNSW : tombstone + eventual rebuild
            self._tombstones.add(int_id)
            self._vectors.pop(int_id, None)
            del self._str_to_int[chunk_id]
            del self._int_to_str[int_id]

            if len(self._tombstones) >= self._rebuild_threshold:
                self._rebuild_hnsw()

    def _rebuild_hnsw(self) -> None:
        """Reconstruit l'index HNSW sans les tombstones."""
        logger.info("Rebuilding HNSW index (%d tombstones)...", len(self._tombstones))
        t0 = time.time()

        new_index = self._create_index()
        if new_index is None:
            return

        # Re-add tous les vecteurs actifs
        active_vecs = {
            iid: vec for iid, vec in self._vectors.items()
            if iid not in self._tombstones
        }

        if active_vecs:
            ids_arr = np.array(list(active_vecs.keys()), dtype=np.int64)
            vecs_arr = np.vstack(list(active_vecs.values())).astype(np.float32)
            new_index.add_with_ids(vecs_arr, ids_arr)

        self.index = new_index
        self._tombstones.clear()
        # Nettoyage _vectors : garder seulement les actifs
        self._vectors = active_vecs

        logger.info("HNSW rebuild done in %.2fs (%d vectors)", time.time() - t0, len(active_vecs))

    def _safe_path(self, path: str) -> str:
        """Remplace les caractères invalides dans les noms de fichiers."""
        # Remplacer '*' par 'star' (problème sur Windows)
        return path.replace('*', 'star')

    def save(self, path: str) -> None:
        if self.index is None:
            return
        safe_path = self._safe_path(path)
        faiss.write_index(self.index, f"{safe_path}.index")
        with open(f"{safe_path}.ids", "wb") as f:
            pickle.dump({
                "str_to_int": self._str_to_int,
                "int_to_str": self._int_to_str,
                "next_id": self._next_id,
                "tombstones": self._tombstones,
                "index_type": self._index_type,
                "vectors": self._vectors if self._index_type == "hnsw" else {},
            }, f)

    def load(self, path: str) -> bool:
        safe_path = self._safe_path(path)
        idx_path = f"{safe_path}.index"
        ids_path = f"{safe_path}.ids"
        if not os.path.exists(idx_path) or not os.path.exists(ids_path):
            return False
        try:
            self.index = faiss.read_index(idx_path)
            with open(ids_path, "rb") as f:
                data = pickle.load(f)
            self._str_to_int = data.get("str_to_int", {})
            self._int_to_str = data.get("int_to_str", {})
            self._next_id = data.get("next_id", 0)
            self._tombstones = data.get("tombstones", set())
            self._vectors = data.get("vectors", {})
            return True
        except Exception as exc:
            logger.warning("FAISS load failed from %s: %s", path, exc)
            return False


# ======================================================================
# LanguageBM25Index — one BM25 corpus per language
# ----------------------------------------------------------------------
# v3.1: add_many() for bulk ingestion + amortized rebuild.
# ======================================================================

class LanguageBM25Index:
    """
    A per-language BM25 corpus.

    Design
    ------
    BM25Okapi (from rank_bm25) precomputes `doc_freqs`, `idf`, `doc_len`
    and `avgdl` at __init__ time. That means it does NOT support true
    incremental updates — any new document can shift IDFs.

    Strategy used here:
      * add()/add_many() only append to the in-memory corpus and flip a
        `_dirty` flag. NO rebuild happens on insert.
      * _rebuild() runs LAZILY on the first search() after any mutation,
        and is further skipped if the corpus size has not changed since
        the last successful build.
      * prewarm() lets ingestion pipelines pay the rebuild cost once at
        the end of ingestion instead of on the first user query.

    For a 500-chunk video this means 1 rebuild instead of 500, and for a
    10k-chunk corpus the rebuild cost is paid exactly once (at prewarm).
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._chunk_ids: List[str] = []
        self._chunk_id_set: Set[str] = set()    # O(1) lookup for dedup
        self._corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = True
        # PERF-BM25-2: track corpus size at last successful build so we
        # can skip redundant rebuilds when _dirty was flipped but the
        # corpus didn't actually grow (duplicate adds, no-op removes).
        self._last_built_size: int = 0
        self._build_count: int = 0

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    @property
    def is_dirty(self) -> bool:
        """True if a rebuild would be needed before the next search."""
        return self._dirty and len(self._corpus) != self._last_built_size

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, chunk_id: str, text: str) -> None:
        """
        Add ONE document. O(tokenize) — no rebuild is triggered.

        For bulk ingestion prefer `add_many()` to avoid repeated
        set/list growth overhead.
        """
        if chunk_id in self._chunk_id_set:
            return  # dedup
        tokens = tokenize(text, self.language)
        if not tokens:
            return
        self._chunk_ids.append(chunk_id)
        self._chunk_id_set.add(chunk_id)
        self._corpus.append(tokens)
        self._dirty = True

    def add_many(self, items: List[Tuple[str, str]]) -> int:
        """
        PERF-BM25-1: bulk add. `items` is a list of (chunk_id, text).

        Performs ALL tokenization and list extension in a single pass,
        flips `_dirty` exactly once, and returns the number of docs
        actually appended (after dedup / empty-tokens filter).

        NO BM25 rebuild is triggered here — it stays lazy.
        """
        if not items:
            return 0
        added = 0
        # Local refs for speed (hot loop)
        cid_set = self._chunk_id_set
        cid_list = self._chunk_ids
        corpus = self._corpus
        lang = self.language
        for chunk_id, text in items:
            if not chunk_id or chunk_id in cid_set:
                continue
            tokens = tokenize(text, lang)
            if not tokens:
                continue
            cid_list.append(chunk_id)
            cid_set.add(chunk_id)
            corpus.append(tokens)
            added += 1
        if added:
            self._dirty = True
        return added

    def remove(self, chunk_id: str) -> None:
        if chunk_id not in self._chunk_id_set:
            return
        idx = self._chunk_ids.index(chunk_id)
        self._chunk_ids.pop(idx)
        self._chunk_id_set.discard(chunk_id)
        self._corpus.pop(idx)
        self._dirty = True

    # ------------------------------------------------------------------
    # Build / prewarm
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """
        Rebuild the BM25Okapi index from the current corpus.

        Short-circuits if:
          * not dirty, OR
          * corpus is empty, OR
          * corpus size is the same as the last successful build
            (PERF-BM25-2 — avoids redundant rebuilds when _dirty was
            spuriously flipped).
        """
        if not self._dirty:
            return
        n = len(self._corpus)
        if n == 0:
            return
        if not BM25_AVAILABLE:
            return
        # Skip if nothing actually changed since last build
        if self._bm25 is not None and n == self._last_built_size:
            self._dirty = False
            return

        k1 = _cfg_float("bm25_k1", 1.5)
        b = _cfg_float("bm25_b", 0.75)

        t0 = time.time()
        self._bm25 = BM25Okapi(self._corpus, k1=k1, b=b)
        elapsed = time.time() - t0
        self._dirty = False
        self._last_built_size = n
        self._build_count += 1

        # PERF-BM25-3: visible log for slow large-corpus rebuilds.
        log_fn = logger.info if elapsed >= 0.5 else logger.debug
        log_fn(
            "BM25 rebuild lang=%s | docs=%d | %.2fs | build#=%d",
            self.language, n, elapsed, self._build_count,
        )

    def prewarm(self) -> None:
        """
        PERF-BM25-4: force a rebuild NOW (if dirty).

        Useful at the end of an ingestion batch so the first user query
        doesn't pay the full rebuild latency. Safe to call multiple times;
        subsequent calls are no-ops when the corpus hasn't grown.
        """
        self._rebuild()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        if not BM25_AVAILABLE or not self._corpus:
            return []
        # Lazy rebuild — amortizes cost to 1x per search burst.
        self._rebuild()
        if self._bm25 is None:
            return []

        tokens = tokenize(query, self.language)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _safe_path(self, path: str) -> str:
        """Remplace les caractères invalides dans les noms de fichiers."""
        return path.replace('*', 'star')

    def save(self, path: str) -> None:
        safe_path = self._safe_path(path)
        with open(safe_path, "wb") as f:
            pickle.dump({
                "language": self.language,
                "chunk_ids": self._chunk_ids,
                "corpus": self._corpus,
            }, f)

    def load(self, path: str) -> bool:
        safe_path = self._safe_path(path)
        if not os.path.exists(safe_path):
            return False
        try:
            with open(safe_path, "rb") as f:
                data = pickle.load(f)
            self.language = data["language"]
            self._chunk_ids = data["chunk_ids"]
            self._chunk_id_set = set(self._chunk_ids)
            self._corpus = data["corpus"]
            # Loaded corpus must be rebuilt before first search.
            self._bm25 = None
            self._dirty = True
            self._last_built_size = 0
            return True
        except Exception as exc:
            logger.warning("BM25 load failed from %s: %s", path, exc)
            return False


# ======================================================================
# VideoStore — FAISS + BM25 hybrid storage
# ======================================================================

class VideoStore:
    """
    In-process FAISS + BM25 hybrid store.
    Per-language FAISS indexes + BM25 corpora.
    Thread-safe with a lock on write operations.

    Public API (unchanged):
      add_embedded_chunks(chunks)
      search(query_embedding, ...)
      list_videos()
      get_stats(video_id=None)
      delete_video(video_id)
      save()
    """

    def __init__(self) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu required: pip install faiss-cpu")

        self.dim: int = _cfg_int("bge_dense_dim", 1024)
        self.store_dir: Path = Path(_cfg("storage_dir", "data/faiss_store"))
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.lang_indexes: List[str] = _cfg_list("faiss_language_indexes", ["ar", "en", "fr"])
        index_type: str = _cfg("faiss_index_type", "flat")
        hnsw_m: int = _cfg_int("faiss_hnsw_m", 32)
        ef_con: int = _cfg_int("faiss_hnsw_ef_construction", 200)
        rebuild_th: int = _cfg_int("faiss_rebuild_threshold", 100)

        self._lock = threading.Lock()

        # Per-language FAISS indexes
        self._faiss: Dict[str, LanguageFAISSIndex] = {}
        for lang in self.lang_indexes:
            self._faiss[lang] = LanguageFAISSIndex(
                self.dim, index_type, hnsw_m, ef_con, rebuild_th,
            )
        self._faiss["*"] = LanguageFAISSIndex(self.dim, "flat")  # cross-language fallback

        # Per-language BM25
        self._bm25: Dict[str, LanguageBM25Index] = {}
        for lang in self.lang_indexes:
            self._bm25[lang] = LanguageBM25Index(lang)
        self._bm25["*"] = LanguageBM25Index("en")

        # Metadata store : chunk_id → payload dict
        self._meta: Dict[str, Dict[str, Any]] = {}
        # Video → chunk_ids
        self._video_chunks: Dict[str, Set[str]] = {}

        # Score cache
        self._cache = ScoreCache(
            max_size=_cfg_int("score_cache_size", 500),
            ttl=_cfg_float("score_cache_ttl", 1800.0),
        )

        # RRF parameter
        self._rrf_k: int = _cfg_int("rrf_k", 60)

        # Snapshot
        self._chunks_since_save: int = 0
        self._snapshot_interval: int = _cfg_int("snapshot_interval_chunks", 50)
        self._auto_save: bool = _cfg_bool("snapshot_auto_save", True)

        # Load existing snapshot
        if self._load_snapshot():
            total = sum(idx.size for idx in self._faiss.values())
            logger.info(
                "VideoStore: snapshot loaded | %d chunks | %d videos",
                len(self._meta), len(self._video_chunks),
            )
        else:
            logger.info("VideoStore: fresh start (no snapshot)")

        logger.info(
            "VideoStore: FAISS+BM25 ready | langs=%s | dim=%d | index=%s",
            self.lang_indexes, self.dim, index_type,
        )

    # ==================================================================
    # Language routing
    # ==================================================================

    def _route_lang(self, language: str) -> str:
        lang = (language or "").strip().lower()[:2]
        return lang if lang in self.lang_indexes else "*"

    # ==================================================================
    # Safe attribute access on chunks
    # ==================================================================

    @staticmethod
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        """getattr qui marche aussi sur dict."""
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # ==================================================================
    # Insert
    # ==================================================================

    def add_embedded_chunks(
        self,
        chunks: list,
        save: bool = False,
        prewarm_bm25: bool = False,
    ) -> int:
        """
        Ajoute des chunks embedés.

        Signature compatible :
            add_embedded_chunks(chunks)
            add_embedded_chunks(chunks, save=True)
            add_embedded_chunks(chunks, save=True, prewarm_bm25=True)

        Le video_id est lu depuis chaque chunk.
        `save=True` force une persistance immédiate du snapshot, ce qui évite
        de perdre une vidéo fraîchement ingérée si le serveur redémarre avant
        d'atteindre `snapshot_interval_chunks`.

        PERF-BM25-1  `prewarm_bm25=True` builds the BM25 index NOW (once
            per touched language) instead of lazily on the first user
            query. Strongly recommended at the END of an ingestion batch.

        PERF-BM25-1  BM25 inserts are BATCHED per-language via add_many(),
            so the expensive BM25Okapi rebuild is never triggered during
            this method — it stays lazy (or is paid once if prewarm_bm25).
        """
        if not chunks:
            return 0

        # PERF-BM25-1: per-language buckets for bulk BM25 insertion.
        # key = bm25 language key (e.g. "en", "fr", "ar", "*")
        # val = list of (chunk_id, text) tuples
        bm25_batches: Dict[str, List[Tuple[str, str]]] = {}

        count = 0
        with self._lock:
            for chunk in chunks:
                dense = self._get(chunk, "bge_dense")
                if dense is None or (isinstance(dense, (list, np.ndarray)) and len(dense) == 0):
                    continue

                chunk_id = str(self._get(chunk, "chunk_id", ""))
                if not chunk_id:
                    continue

                # Dedup
                if chunk_id in self._meta:
                    continue

                video_id = str(self._get(chunk, "video_id", ""))
                language = str(self._get(chunk, "language", "unknown"))
                text = str(self._get(chunk, "text", ""))
                search_text = str(self._get(chunk, "search_text", "") or text)
                original_text = str(self._get(chunk, "original_text", "") or text)

                lang_key = self._route_lang(language)

                # FAISS
                vec = np.array(dense, dtype=np.float32)
                self._faiss[lang_key].add(chunk_id, vec)
                if lang_key != "*":
                    self._faiss["*"].add(chunk_id, vec)

                # BM25  (PERF-BM25-1: accumulate, flush once at end)
                bm25_key = lang_key if lang_key in self._bm25 else "*"
                bm25_batches.setdefault(bm25_key, []).append(
                    (chunk_id, search_text or text)
                )
                # Metadata — include filter_tags from chunk metadata
                chunk_meta = self._get(chunk, "metadata", {}) or {}
                clip_embedding = (
                    self._get(chunk, "clip_embedding")
                    or chunk_meta.get("clip_embedding")
                    or self._get(chunk, "embedding")
                )
                clip_embedding_dim = (
                    self._get(chunk, "clip_embedding_dim")
                    or chunk_meta.get("clip_embedding_dim")
                    or self._get(chunk, "embedding_dim", 0)
                )
                if clip_embedding is not None and "clip_embedding" not in chunk_meta:
                    chunk_meta["clip_embedding"] = clip_embedding
                if clip_embedding_dim and "clip_embedding_dim" not in chunk_meta:
                    try:
                        chunk_meta["clip_embedding_dim"] = int(clip_embedding_dim)
                    except Exception:
                        chunk_meta["clip_embedding_dim"] = 0
                self._meta[chunk_id] = {
                    "chunk_id": chunk_id,
                    "video_id": video_id,
                    "text": text,
                    "original_text": original_text,
                    "search_text": search_text,
                    "start_time": self._get(chunk, "start_time", 0.0),
                    "end_time": self._get(chunk, "end_time", 0.0),
                    "language": language,
                    "speaker": str(self._get(chunk, "speaker", "SPEAKER_00") or "SPEAKER_00"),
                    "topic": str(self._get(chunk, "topic", "general")),
                    "chunk_type": str(self._get(chunk, "chunk_type", "speech")),
                    "key_points": self._get(chunk, "key_points", []) or [],
                    "clip_embedding": chunk_meta.get("clip_embedding"),
                    "clip_embedding_dim": chunk_meta.get("clip_embedding_dim", 0),
                    "metadata": chunk_meta,
                }

                # Track video → chunks
                if video_id not in self._video_chunks:
                    self._video_chunks[video_id] = set()
                self._video_chunks[video_id].add(chunk_id)

                count += 1

            # PERF-BM25-1: flush accumulated BM25 batches per-language.
            # This is ONE _dirty flip per language, not one per chunk.
            for bm25_key, items in bm25_batches.items():
                if items:
                    self._bm25[bm25_key].add_many(items)

            # PERF-BM25-4: optional eager rebuild so the first user query
            # doesn't pay the full build latency.
            if prewarm_bm25 and count > 0:
                touched_langs = list(bm25_batches.keys())
                for bm25_key in touched_langs:
                    try:
                        self._bm25[bm25_key].prewarm()
                    except Exception as exc:
                        logger.warning(
                            "BM25 prewarm failed lang=%s: %s", bm25_key, exc
                        )

        self._chunks_since_save += count

        # Persistance explicite : utilisée par l'ingestion pour garantir que
        # la vidéo survive à un refresh / restart immédiat.
        if count > 0 and save:
            self._save_snapshot()
            self._chunks_since_save = 0
        # Auto-snapshot périodique pour limiter les écritures disque pendant
        # les batchs volumineux.
        elif self._auto_save and self._chunks_since_save >= self._snapshot_interval:
            self._save_snapshot()
            self._chunks_since_save = 0

        logger.info("VideoStore: stored %d chunks (total: %d)", count, len(self._meta))
        return count

    # ==================================================================
    # Search — FAISS dense + BM25 sparse → RRF fusion
    # ==================================================================

    def search(
        self,
        query_embedding: Any = None,
        query: str = "",
        limit: int = 10,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        with_vision: bool = True,
        chunk_type: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
        filter_tags: Optional[List[str]] = None,   # DOC7-FIX-2 : schéma v2.0
        **kwargs,
    ) -> List[SearchResult]:
        effective_limit = top_k if top_k is not None else limit
        if effective_limit <= 0:
            return []

        # Extract text & vectors from query_embedding
        query_text = ""
        dense_vec = None

        if query_embedding is not None:
            query_text = str(self._get(query_embedding, "text", "")) or ""
            expanded = self._get(query_embedding, "expanded_text")
            if expanded:
                query_text = f"{query_text} {expanded}"
            dense_vec = self._get(query_embedding, "bge_dense")

        # Fallback: use query parameter as text
        if not query_text and query:
            query_text = query

        if dense_vec is None or (isinstance(dense_vec, (list, np.ndarray)) and len(dense_vec) == 0):
            logger.warning("search: no dense vector available")
            return []

        # Check cache
        cached = self._cache.get(query_text, video_id)
        if cached is not None:
            return cached[:effective_limit]

        qvec = np.array(dense_vec, dtype=np.float32)
        candidate_k = max(effective_limit * 5, 100)

        # Detect query language
        query_lang = language or detect_language(query_text)
        lang_key = self._route_lang(query_lang)
        dense_w, sparse_w = _get_fusion_weights(query_lang)

        # --- FAISS dense search ---
        dense_results: List[Tuple[str, float]] = []
        if lang_key != "*":
            dense_results = self._faiss[lang_key].search(qvec, candidate_k)
        fallback_dense = self._faiss["*"].search(qvec, candidate_k)
        seen_dense = set(cid for cid, _ in dense_results)
        for cid, score in fallback_dense:
            if cid not in seen_dense:
                dense_results.append((cid, score))
                seen_dense.add(cid)

        # --- BM25 sparse search ---
        sparse_results: List[Tuple[str, float]] = []
        if BM25_AVAILABLE and query_text:
            bm25_key = lang_key if lang_key in self._bm25 else "*"
            sparse_results = self._bm25[bm25_key].search(query_text, candidate_k)
            if bm25_key != "*":
                fb_sparse = self._bm25["*"].search(query_text, candidate_k)
                seen_sparse = set(cid for cid, _ in sparse_results)
                for cid, score in fb_sparse:
                    if cid not in seen_sparse:
                        sparse_results.append((cid, score))

        # --- Filtrage (DOC7-FIX-2 : filter_tags support) ---
        def _passes(cid: str) -> bool:
            meta = self._meta.get(cid, {})
            if video_id and meta.get("video_id") != video_id:
                return False
            if speaker and meta.get("speaker") != speaker:
                return False
            if not with_vision and meta.get("chunk_type") == "visual":
                return False
            if chunk_type and meta.get("chunk_type") != chunk_type:
                return False
            if chunk_types and meta.get("chunk_type") not in chunk_types:
                return False
            # DOC7-FIX-2 : filter_tags (schéma v2.0)
            if filter_tags:
                chunk_tags = meta.get("metadata", {}).get("filter_tags", [])
                if not any(tag in chunk_tags for tag in filter_tags):
                    return False
            return True

        dense_results = [(c, s) for c, s in dense_results if _passes(c)]
        sparse_results = [(c, s) for c, s in sparse_results if _passes(c)]

        # --- RRF fusion ---
        fused = self._rrf_fuse(
            dense_results, sparse_results, dense_w, sparse_w, effective_limit,
        )

        # Convert to SearchResult
        results = [self._to_search_result(cid, score, ds, ss, strat)
                   for cid, score, ds, ss, strat in fused]

        # Cache
        self._cache.put(query_text, video_id, results)

        logger.info(
            "VideoStore.search → %d results (dense=%d sparse=%d lang=%s)",
            len(results), len(dense_results), len(sparse_results), query_lang,
        )
        return results

    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense: List[Tuple[str, float]],
        sparse: List[Tuple[str, float]],
        dense_w: float,
        sparse_w: float,
        limit: int,
    ) -> List[Tuple[str, float, float, float, str]]:
        """
        RRF with language-adaptive weights.
        CRIT-4 : dense_score et sparse_score sont TOUJOURS float, jamais None.
        DOC7-FIX-1 : track dense_set / sparse_set explicitly to avoid misclassifying
                     cosine_score == 0.0 as "not from dense".
        Returns (chunk_id, fused_score, dense_score, sparse_score, strategy).
        """
        scores: Dict[str, Dict[str, float]] = {}
        # DOC7-FIX-1 : explicit source sets
        dense_set: Set[str] = set()
        sparse_set: Set[str] = set()

        for rank, (cid, raw) in enumerate(dense):
            if cid not in scores:
                scores[cid] = {"fused": 0.0, "dense": 0.0, "sparse": 0.0}
            scores[cid]["fused"] += dense_w / (self._rrf_k + rank + 1)
            scores[cid]["dense"] = float(raw)
            dense_set.add(cid)

        for rank, (cid, raw) in enumerate(sparse):
            if cid not in scores:
                scores[cid] = {"fused": 0.0, "dense": 0.0, "sparse": 0.0}
            scores[cid]["fused"] += sparse_w / (self._rrf_k + rank + 1)
            scores[cid]["sparse"] = float(raw)
            sparse_set.add(cid)

        ranked = sorted(scores.items(), key=lambda x: x[1]["fused"], reverse=True)

        results: List[Tuple[str, float, float, float, str]] = []
        for cid, d in ranked[:limit]:
            # DOC7-FIX-1 : use explicit sets instead of score > 0 check
            in_dense = cid in dense_set
            in_sparse = cid in sparse_set
            if in_dense and in_sparse:
                strat = "hybrid_rrf"
            elif in_dense:
                strat = "dense_only"
            elif in_sparse:
                strat = "sparse_only"
            else:
                strat = "unknown"
            results.append((cid, d["fused"], d["dense"], d["sparse"], strat))

        return results

    # ------------------------------------------------------------------

    def _to_search_result(
        self,
        chunk_id: str,
        score: float,
        dense_score: float,
        sparse_score: float,
        strategy: str,
    ) -> SearchResult:
        meta = self._meta.get(chunk_id, {})
        return SearchResult(
            chunk_id=chunk_id,
            video_id=str(meta.get("video_id", "")),
            text=meta.get("text", ""),
            score=score,
            start_time=meta.get("start_time"),
            end_time=meta.get("end_time"),
            language=meta.get("language"),
            speaker=meta.get("speaker"),
            topic=meta.get("topic"),
            chunk_type=meta.get("chunk_type", "speech"),
            strategy=strategy,
            dense_score=dense_score,      # CRIT-4 : toujours float
            sparse_score=sparse_score,    # CRIT-4 : toujours float
            key_points=meta.get("key_points", []),
            metadata=meta.get("metadata", {}),
        )

    # ==================================================================
    # Video management
    # ==================================================================

    def get_chunks_by_time_range(
        self,
        video_id: str,
        start_seconds: float,
        end_seconds: float,
        chunk_types: Optional[List[str]] = None,
    ) -> List["SearchResult"]:
        """
        Return all chunks whose time window overlaps [start_seconds, end_seconds].
        This is a direct metadata scan — no vector search required.
        Used for timestamp_lookup queries so we return the ACTUAL content at that moment.
        """
        chunk_ids = self._video_chunks.get(video_id, set())
        results = []
        for cid in chunk_ids:
            meta = self._meta.get(cid, {})
            c_start = float(meta.get("start_time") or 0.0)
            c_end = float(meta.get("end_time") or c_start + 30.0)
            if chunk_types and meta.get("chunk_type") not in chunk_types:
                continue
            # overlap check
            if c_start <= end_seconds and c_end >= start_seconds:
                results.append(self._to_search_result(cid, score=1.0,
                                                  dense_score=0.0, sparse_score=0.0,
                                                  strategy="time_range"))
        results.sort(key=lambda r: (r.start_time or 0.0))
        return results

    def get_all_chunks_for_video(
        self,
        video_id: str,
        chunk_types: Optional[List[str]] = None,
        speech_only: bool = False,
    ) -> List["SearchResult"]:
        """
        Return ALL stored chunks for a video, sorted chronologically.
        Used for yes_no broad-evidence queries so the LLM sees the full transcript.
        """
        chunk_ids = self._video_chunks.get(video_id, set())
        results = []
        for cid in chunk_ids:
            meta = self._meta.get(cid, {})
            if speech_only and meta.get("chunk_type", "speech") != "speech":
                continue
            if chunk_types and meta.get("chunk_type") not in chunk_types:
                continue
            results.append(self._to_search_result(cid, score=1.0,
                                              dense_score=0.0, sparse_score=0.0,
                                              strategy="full_scan"))
        results.sort(key=lambda r: (r.start_time or 0.0))
        return results

    def fuzzy_text_search(
        self,
        video_id: str,
        tokens: List[str],
        top_k: int = 5,
        min_token_len: int = 4,
    ) -> List["SearchResult"]:
        """
        Phonetic / fuzzy substring search over all stored chunk texts for a video.

        This is the STT-error fallback: when a yes/no query contains a name or
        entity that doesn't appear literally in any retrieved chunk, we scan all
        chunks for phonetically similar substrings.

        Strategy (no external library needed):
        1. For each query token (length >= min_token_len), generate candidate
           prefixes and suffixes (first 4 chars, last 4 chars).
        2. Score each chunk by how many of those sub-patterns appear in its text.
        3. Return top_k chunks sorted by score descending.

        This catches corruption like "Geoffrey" → "Jeffington" because both share
        the substring "jeff" / "eff" and the 4-char prefix overlap.
        """
        chunk_ids = self._video_chunks.get(video_id, set())
        if not chunk_ids or not tokens:
            return []

        # Build match patterns: for each token, take 4-char prefix and 4-char suffix
        patterns: List[str] = []
        for tok in tokens:
            t = tok.lower().strip(".,!?\"'")
            if len(t) < min_token_len:
                continue
            patterns.append(t[:4])           # first 4 chars
            if len(t) >= 6:
                patterns.append(t[-4:])      # last 4 chars
            patterns.append(t)               # exact (may appear verbatim)

        if not patterns:
            return []

        # Deduplicate patterns
        patterns = list(dict.fromkeys(patterns))

        scored: List[tuple] = []
        for cid in chunk_ids:
            meta = self._meta.get(cid, {})
            if meta.get("chunk_type", "speech") != "speech":
                continue
            text = (meta.get("text") or meta.get("search_text") or "").lower()
            if not text:
                continue
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                scored.append((cid, score))

        scored.sort(key=lambda x: -x[1])
        results = []
        for cid, score in scored[:top_k]:
            results.append(self._to_search_result(
                cid,
                score=min(1.0, score / max(len(patterns), 1)),
                dense_score=0.0,
                sparse_score=0.0,
                strategy="fuzzy_text",
            ))
        return results

    def list_videos(self, limit: int = 1000) -> List[str]:
        """Retourne List[str] de video_ids — compatible api/main.py."""
        result = sorted(self._video_chunks.keys())
        logger.info("list_videos: %d videos", len(result))
        return result[:limit]

    def get_stats(self, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        CRIT-5 : champ unifié "total_chunks" pour global ET per-video.
        DOC7-FIX-3 : ajout filter_tags dans les stats per-video.
        """
        if video_id:
            chunk_ids = self._video_chunks.get(video_id, set())
            # Collecter langues et speakers
            languages: Set[str] = set()
            speakers: Set[str] = set()
            all_tags: Set[str] = set()  # DOC7-FIX-3

            for cid in chunk_ids:
                meta = self._meta.get(cid, {})
                lang = meta.get("language")
                spk = meta.get("speaker")
                if lang and lang != "unknown":
                    languages.add(lang)
                if spk and spk != "unknown":
                    speakers.add(spk)
                # DOC7-FIX-3 : collect filter_tags
                tags = meta.get("metadata", {}).get("filter_tags", [])
                all_tags.update(tags)

            return {
                "video_id": video_id,
                "total_chunks": len(chunk_ids),    # CRIT-5 : unifié
                "chunks": len(chunk_ids),           # backward compat
                "languages": sorted(languages),
                "speakers": sorted(speakers),
                "filter_tags": sorted(all_tags),    # DOC7-FIX-3
            }

        # Global stats
        faiss_total = sum(idx.size for idx in self._faiss.values())
        return {
            "total_chunks": len(self._meta),
            "total_faiss_vectors": faiss_total,
            "total_videos": len(self._video_chunks),
            "languages": list(self._faiss.keys()),
            "bm25_corpora": {k: v.size for k, v in self._bm25.items()},
            "cache": {"hits": self._cache.hits, "misses": self._cache.misses},
        }

    def delete_video(self, video_id: str) -> None:
        """Supprime un video et tous ses chunks de FAISS, BM25, et meta."""
        with self._lock:
            chunk_ids = self._video_chunks.pop(video_id, set())
            for cid in chunk_ids:
                meta = self._meta.pop(cid, {})
                lang_key = self._route_lang(meta.get("language", ""))

                # FAISS : vraie suppression
                if lang_key in self._faiss:
                    self._faiss[lang_key].remove(cid)
                self._faiss["*"].remove(cid)

                # BM25
                bm25_key = lang_key if lang_key in self._bm25 else "*"
                self._bm25[bm25_key].remove(cid)

        self._cache.invalidate_video(video_id)
        logger.info("VideoStore: deleted %d chunks for '%s'", len(chunk_ids), video_id)

        if self._auto_save:
            self._save_snapshot()

    # ==================================================================
    # Snapshot engine
    # ==================================================================

    def _save_snapshot(self) -> None:
        t0 = time.time()
        base = str(self.store_dir)

        # FAISS - utiliser _safe_path via les indexes
        for lang, idx in self._faiss.items():
            idx.save(os.path.join(base, f"faiss_{lang}"))

        # BM25 - utiliser _safe_path via les indexes
        for lang, bm in self._bm25.items():
            bm.save(os.path.join(base, f"bm25_{lang}.pkl"))

        # Metadata
        with open(os.path.join(base, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "meta": self._meta,
                "video_chunks": {k: list(v) for k, v in self._video_chunks.items()},
            }, f)

        logger.info(
            "VideoStore: snapshot saved in %.2fs (%d chunks, %d videos)",
            time.time() - t0, len(self._meta), len(self._video_chunks),
        )

    def _load_snapshot(self) -> bool:
        base = str(self.store_dir)
        meta_path = os.path.join(base, "metadata.pkl")
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self._meta = data.get("meta", {})
            vc = data.get("video_chunks", {})
            self._video_chunks = {k: set(v) for k, v in vc.items()}

            for lang in list(self._faiss.keys()):
                self._faiss[lang].load(os.path.join(base, f"faiss_{lang}"))

            for lang in list(self._bm25.keys()):
                self._bm25[lang].load(os.path.join(base, f"bm25_{lang}.pkl"))

            return True
        except Exception as exc:
            logger.warning("Snapshot load failed: %s", exc)
            return False

    def save(self) -> None:
        """Public trigger for manual snapshot."""
        self._save_snapshot()