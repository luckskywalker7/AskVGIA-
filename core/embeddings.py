"""
core/embeddings.py — Multi-vector embedding pipeline v3.0
==========================================================
Corrections v2.0 → v3.0 :

 BUG-1  _lazy_init_nomic() : si NomicEngine.load() échoue, self._available=False
        mais aucun log n'est émis lors des tentatives suivantes d'encode.
        FIX : ajouter un flag _nomic_init_attempted pour ne loguer qu'une fois,
        et un log explicite "Nomic disabled" si load échoue.

 BUG-2  EmbeddingCache : TTL 1h → ré-encode après 1h. Pour un système de
        production, persister le cache entre les redémarrages.
        FIX : sauvegarder/charger le cache depuis un fichier pickle.
        Les chunks ingérés ne changent pas → cache persisté = pas de ré-encode.

 BUG-3  QueryExpander._check_ollama() synchrone dans __init__ → bloque
        l'init de EmbeddingGenerator si Ollama est lent.
        FIX : lazy check au premier appel de expand(), pas dans __init__.

 BONUS  sha256 au lieu de md5 dans EmbeddingCache (cohérent avec storage v3).
        Safe settings access. Logging amélioré partout.

4-vector fusion with language-adaptive routing :
  1. BGE-M3 dense (1024d) — semantic backbone
  2. BGE-M3 sparse (lexical) — Arabic morphology precision
  3. Nomic-embed-text (768d, matryoshka 256d) — fast first-pass (optional)
  4. BGE-M3 ColBERT (token-level) — late interaction rerank (optional)

All engines OPTIONAL — graceful degradation to BGE-only.

Public API unchanged :
  embed_query(text) → QueryEmbedding
  generate_chunks(chunks, video_id) → List[EmbeddedChunk]
  embed(chunks, video_id) → alias
  embed_chunks(chunks, video_id) → alias
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ══════════════════════════════════════════════════════════════════════
# Data contracts
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SparseVectorData:
    indices: List[int]
    values: List[float]


@dataclass
class QueryEmbedding:
    text: str
    bge_dense: List[float]
    bge_sparse: Optional[SparseVectorData] = None
    nomic_dense: Optional[List[float]] = None
    colbert_query: Optional[List[List[float]]] = None
    expanded_text: Optional[str] = None


@dataclass
class EmbeddedChunk:
    chunk_id: str
    video_id: str
    text: str
    original_text: str
    search_text: str
    start_time: float
    end_time: float
    language: str = "unknown"
    speaker: str = "unknown"
    topic: str = "general"
    chunk_type: str = "speech"
    bge_dense: Optional[List[float]] = None
    bge_sparse: Optional[SparseVectorData] = None
    nomic_dense: Optional[List[float]] = None
    colbert_vecs: Optional[List[List[float]]] = None
    key_points_vec: Optional[List[float]] = None
    key_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _safe_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """getattr qui marche aussi sur dict."""
    if isinstance(obj, dict):
        for name in names:
            val = obj.get(name)
            if val is not None:
                return val
    else:
        for name in names:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _deterministic_id(video_id: str, text: str, start: float, end: float) -> str:
    raw = f"{video_id}|{start:.3f}|{end:.3f}|{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ══════════════════════════════════════════════════════════════════════
# BUG-2 FIX : EmbeddingCache — LRU + TTL + persistence
# ══════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    LRU cache for embeddings keyed on sha256(text).

    BUG-2 FIX : persistance entre redémarrages via pickle.
    Les chunks ingérés ne changent pas → le cache persiste = pas de ré-encode.

    BONUS : sha256 au lieu de md5 (cohérent avec storage v3).
    Thread-safe avec un lock.
    """

    def __init__(
        self,
        max_size: int = 2048,
        ttl_seconds: float = 86400.0,      # 24h au lieu de 1h
        persist_path: Optional[str] = None,
    ):
        self._cache: OrderedDict[str, Tuple[float, str, Dict[str, Any]]] = OrderedDict()
        # value = (timestamp, raw_text_for_collision_check, data)
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        # BUG-2 FIX : persistence
        self._persist_path = persist_path or str(
            Path(_cfg("storage_dir", "data/faiss_store")) / "embedding_cache.pkl"
        )
        self._dirty = False
        self._save_interval = 50   # save every N puts

        self._load_from_disk()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        k = self._key(text)
        with self._lock:
            if k in self._cache:
                ts, stored_text, data = self._cache[k]
                # Anti-collision : vérifier le texte brut
                if stored_text != text:
                    self._misses += 1
                    return None
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(k)
                    self._hits += 1
                    return data
                # Expired
                del self._cache[k]
            self._misses += 1
            return None

    def put(self, text: str, data: Dict[str, Any]) -> None:
        k = self._key(text)
        with self._lock:
            self._cache[k] = (time.time(), text, data)
            self._cache.move_to_end(k)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            self._dirty = True

        # Periodic persistence
        if self._dirty and (self._hits + self._misses) % self._save_interval == 0:
            self._save_to_disk()

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

    # ── Persistence ──────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        """BUG-2 FIX : persister le cache sur disque."""
        if not self._dirty:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                # Sauvegarder sans le texte brut (économie d'espace)
                # On garde quand même le texte pour la vérification anti-collision
                snapshot = dict(self._cache)

            with open(str(path), "wb") as f:
                pickle.dump({
                    "version": 3,
                    "cache": snapshot,
                    "max_size": self._max_size,
                    "ttl": self._ttl,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            self._dirty = False
            logger.debug("EmbeddingCache saved: %d entries → %s", len(snapshot), path)
        except Exception as exc:
            logger.warning("EmbeddingCache save failed: %s", exc)

    def _load_from_disk(self) -> None:
        """BUG-2 FIX : charger le cache depuis le disque au démarrage."""
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            with open(str(path), "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict) or data.get("version") != 3:
                logger.info("EmbeddingCache: incompatible version, starting fresh")
                return

            loaded = data.get("cache", {})
            now = time.time()
            ttl = data.get("ttl", self._ttl)

            # Filtrer les entrées expirées
            valid = 0
            expired = 0
            with self._lock:
                for k, (ts, raw_text, entry_data) in loaded.items():
                    if now - ts < ttl:
                        self._cache[k] = (ts, raw_text, entry_data)
                        valid += 1
                    else:
                        expired += 1

            logger.info(
                "EmbeddingCache loaded: %d valid, %d expired, from %s",
                valid, expired, path,
            )
        except Exception as exc:
            logger.warning("EmbeddingCache load failed: %s", exc)

    def save(self) -> None:
        """Public trigger for manual save."""
        self._dirty = True
        self._save_to_disk()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
        self._dirty = True
        self._save_to_disk()


# ══════════════════════════════════════════════════════════════════════
# Language routing — adaptive weights
# ══════════════════════════════════════════════════════════════════════

LANG_WEIGHTS: Dict[str, Dict[str, float]] = {
    "ar": {"bge_dense": 0.30, "bge_sparse": 0.35, "nomic": 0.10, "colbert": 0.25},
    "fa": {"bge_dense": 0.30, "bge_sparse": 0.35, "nomic": 0.10, "colbert": 0.25},
    "ur": {"bge_dense": 0.30, "bge_sparse": 0.35, "nomic": 0.10, "colbert": 0.25},
    "en": {"bge_dense": 0.30, "bge_sparse": 0.15, "nomic": 0.30, "colbert": 0.25},
    "fr": {"bge_dense": 0.30, "bge_sparse": 0.20, "nomic": 0.25, "colbert": 0.25},
}
DEFAULT_WEIGHTS = {"bge_dense": 0.35, "bge_sparse": 0.25, "nomic": 0.15, "colbert": 0.25}

SPARSE_PRUNE_THRESHOLD = 0.01


def get_lang_weights(language: str) -> Dict[str, float]:
    return LANG_WEIGHTS.get(language, DEFAULT_WEIGHTS)


# ══════════════════════════════════════════════════════════════════════
# GPU auto-detection helper
# ══════════════════════════════════════════════════════════════════════

def _auto_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("BGE-M3: CUDA available → using GPU (%s)", torch.cuda.get_device_name(0))
            return "cuda"
    except Exception:
        pass
    logger.info("BGE-M3: No CUDA detected → using CPU")
    return "cpu"


# ══════════════════════════════════════════════════════════════════════
# BGE-M3 Engine (dense + sparse + ColBERT)
# ══════════════════════════════════════════════════════════════════════

class BGEM3Engine:
    """
    BAAI/bge-m3 via FlagEmbedding.
    Dense 1024d + sparse lexical + ColBERT token-level vectors.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
        batch_size: int = 12,
        max_length: int = 8192,
        enable_colbert: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        # If device not explicitly set, read from settings first, then auto-detect
        if device is not None:
            self.device = device
        else:
            cfg_device = _cfg("bge_device", "")
            if cfg_device:
                self.device = cfg_device
                logger.info("BGE-M3: using device from settings: %s", self.device)
            else:
                self.device = _auto_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_colbert = enable_colbert
        self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding not installed: pip install FlagEmbedding"
            ) from exc

        t0 = time.time()
        logger.info(
            "Loading BGE-M3 | model=%s | device=%s | fp16=%s | colbert=%s",
            self.model_name, self.device, self.use_fp16, self.enable_colbert,
        )
        self._model = BGEM3FlagModel(
            self.model_name, use_fp16=self.use_fp16, device=self.device,
        )
        logger.info("BGE-M3 loaded in %.2fs", time.time() - t0)

    def _ensure_loaded(self) -> None:
        if not self.is_loaded():
            self.load()

    def _adaptive_batch_size(self, texts: List[str]) -> int:
        if not texts:
            return self.batch_size
        avg_len = sum(len(t) for t in texts) / len(texts)
        if avg_len > 2000:
            return max(1, self.batch_size // 4)
        if avg_len > 1000:
            return max(2, self.batch_size // 2)
        return self.batch_size

    def encode_batch(
        self, texts: List[str], with_colbert: bool = False,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        texts = [_normalize_text(t) for t in texts if _normalize_text(t)]
        if not texts:
            return []

        actual_batch = self._adaptive_batch_size(texts)
        return_colbert = with_colbert and self.enable_colbert

        all_results: List[Dict[str, Any]] = []

        for offset in range(0, len(texts), actual_batch):
            batch = texts[offset:offset + actual_batch]
            try:
                raw = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=return_colbert,
                )
            except Exception as exc:
                logger.error("BGE-M3 encode failed on batch %d: %s", offset, exc)
                if return_colbert:
                    try:
                        raw = self._model.encode(
                            batch,
                            batch_size=len(batch),
                            max_length=self.max_length,
                            return_dense=True,
                            return_sparse=True,
                            return_colbert_vecs=False,
                        )
                        return_colbert = False
                        logger.info("BGE-M3 fallback: colbert disabled for this batch")
                    except Exception:
                        raise
                else:
                    raise

            dense_vecs = raw.get("dense_vecs", [])
            lexical = raw.get("lexical_weights", [])
            colbert = raw.get("colbert_vecs", []) if return_colbert else []

            for i in range(len(batch)):
                entry: Dict[str, Any] = {
                    "dense": (
                        self._to_list(dense_vecs[i])
                        if i < len(dense_vecs) else []
                    ),
                    "sparse": (
                        self._normalize_sparse(lexical[i])
                        if i < len(lexical) else None
                    ),
                }
                if return_colbert and i < len(colbert):
                    entry["colbert"] = [self._to_list(v) for v in colbert[i]]
                all_results.append(entry)

        return all_results

    def encode_query(
        self, text: str, with_colbert: bool = True,
    ) -> Dict[str, Any]:
        results = self.encode_batch([text], with_colbert=with_colbert)
        return results[0] if results else {"dense": [], "sparse": None}

    def _to_list(self, vec: Any) -> List[float]:
        if vec is None:
            return []
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)

    def _normalize_sparse(self, raw: Any) -> Optional[Dict[int, float]]:
        if raw is None:
            return None
        converted = {}
        items = raw.items() if hasattr(raw, "items") else []
        for k, v in items:
            try:
                idx, val = int(k), float(v)
                if val != 0.0:
                    converted[idx] = val
            except Exception:
                continue
        return converted or None

    def sparse_to_qdrant(self, sparse_dict: Any) -> Optional[SparseVectorData]:
        """Convert sparse dict → SparseVectorData, PRUNING tokens < threshold."""
        normalized = self._normalize_sparse(sparse_dict)
        if not normalized:
            return None
        pruned = {
            k: v for k, v in normalized.items()
            if abs(v) >= SPARSE_PRUNE_THRESHOLD
        }
        if not pruned:
            return None
        items = sorted(pruned.items(), key=lambda x: x[0])
        return SparseVectorData(
            indices=[idx for idx, _ in items],
            values=[val for _, val in items],
        )


# ══════════════════════════════════════════════════════════════════════
# BUG-1 FIX : NomicEngine — with explicit logging on failure
# ══════════════════════════════════════════════════════════════════════

class NomicEngine:
    """
    nomic-embed-text via sentence-transformers.
    Matryoshka truncation: 768d → 256d for fast retrieval.

    BUG-1 FIX : _load_attempted flag + log explicite si load échoue.
    Après un échec de load, ne retente plus et loggue une fois.
    """

    def __init__(self, truncate_dim: int = 256):
        self._model = None
        self._available = False
        self._load_attempted = False       # BUG-1 FIX
        self._load_error: Optional[str] = None  # BUG-1 FIX
        self.truncate_dim = truncate_dim
        self.full_dim = 768

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> None:
        """
        BUG-1 FIX : ne tente le load qu'une seule fois.
        Si le load échoue → log explicite + ne retente plus.
        """
        if self._model is not None:
            return
        if self._load_attempted:
            # Déjà tenté et échoué → ne pas retenter
            return
        self._load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            logger.info("Loading Nomic-embed-text (truncate=%dd)...", self.truncate_dim)
            self._model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True,
            )
            self._available = True
            logger.info("✅ Nomic loaded in %.2fs", time.time() - t0)
        except ImportError:
            self._load_error = "sentence-transformers not installed"
            logger.warning(
                "⚠️  Nomic DISABLED: sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("⚠️  Nomic DISABLED: load failed — %s", exc)

    def encode(self, texts: List[str], truncate: bool = True) -> List[List[float]]:
        if not self._available or not self._model:
            return []
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            result = []
            for emb in embeddings:
                vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                if truncate and len(vec) > self.truncate_dim:
                    vec = vec[:self.truncate_dim]
                    norm = math.sqrt(sum(x * x for x in vec))
                    if norm > 0:
                        vec = [x / norm for x in vec]
                result.append(vec)
            return result
        except Exception as exc:
            logger.warning("Nomic encode failed: %s", exc)
            return []

    def encode_query(self, text: str) -> Optional[List[float]]:
        results = self.encode([f"search_query: {text}"], truncate=True)
        return results[0] if results else None

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self.encode(prefixed, truncate=True)


# ══════════════════════════════════════════════════════════════════════
# BUG-3 FIX : QueryExpander — lazy Ollama check
# ══════════════════════════════════════════════════════════════════════

class QueryExpander:
    """
    Expand short queries using HyDE (Hypothetical Document Embeddings).
    Uses Ollama if available.

    BUG-3 FIX : _check_ollama() est lazy — exécuté au premier appel de
    expand(), pas dans __init__. Timeout réduit à 1.5s.
    """

    def __init__(self):
        self._available: Optional[bool] = None     # None = pas encore vérifié
        self._checked = False                       # BUG-3 FIX
        self._model_name: Optional[str] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._lazy_check_ollama()
        return self._available or False

    def _lazy_check_ollama(self) -> None:
        """
        BUG-3 FIX : vérification lazy au premier appel.
        Timeout court (1.5s) pour ne pas bloquer l'init.
        """
        if self._checked:
            return
        self._checked = True

        try:
            import requests
            r = requests.get(
                "http://localhost:11434/api/tags",
                timeout=1.5,    # BUG-3 : timeout court
            )
            if r.status_code == 200:
                data = r.json()
                models = data.get("models", [])
                # Chercher un petit modèle pour HyDE
                small_models = [
                    m["name"] for m in models
                    if isinstance(m, dict) and "name" in m
                ]
                if small_models:
                    # Préférer les petits modèles
                    for preferred in ("qwen3.5:2b", "qwen3.5", "qwen", "llama3.2:3b", "llama3.2:1b", "phi", "gemma"):
                        for m in small_models:
                            if preferred in m.lower():
                                self._model_name = m
                                break
                        if self._model_name:
                            break
                    if not self._model_name:
                        self._model_name = small_models[0]

                    self._available = True
                    logger.info("✅ QueryExpander (HyDE): model=%s", self._model_name)
                else:
                    self._available = False
                    logger.info("QueryExpander: Ollama OK but no models installed")
            else:
                self._available = False
                logger.debug("QueryExpander: Ollama returned status %d", r.status_code)
        except ImportError:
            self._available = False
            logger.debug("QueryExpander: requests not installed")
        except Exception as exc:
            self._available = False
            logger.debug("QueryExpander: Ollama not reachable — %s", exc)

    # Language-specific HyDE prompt templates
    _HYDE_PROMPTS = {
        "ar": (
            "بناءً على هذا السؤال حول فيديو، اكتب جملتين مفيدتين "
            "كما قد تظهران في نص الفيديو أو ترجمته. "
            "اكتب باللغة العربية فقط.\n"
            "السؤال: {query}\nالإجابة الافتراضية:"
        ),
        "fr": (
            "Sur la base de cette question à propos d'une vidéo, écris 2 phrases "
            "qui pourraient apparaître dans la transcription. Sois factuel et concis.\n"
            "Question: {query}\nRéponse hypothétique:"
        ),
        "en": (
            "Given this question about a video, write a 2-sentence "
            "hypothetical answer that might appear in the video transcript. "
            "Be factual and concise.\n"
            "Question: {query}\nHypothetical answer:"
        ),
    }

    @staticmethod
    def _detect_query_language(query: str) -> str:
        """Fast language detection from script characters."""
        ar_chars = sum(1 for c in query if "\u0600" <= c <= "\u06FF")
        if ar_chars >= max(1, len(query.replace(" ", "")) * 0.15):
            return "ar"
        fr_markers = ("est-ce", "qu'il", "qu'elle", " les ", " des ", " une ", " que ", " qui ")
        if any(m in query.lower() for m in fr_markers):
            return "fr"
        return "en"

    def expand(self, query: str, language: Optional[str] = None) -> Optional[str]:
        """Generate a hypothetical answer snippet for the query in the same language."""
        if not self.available:
            return None
        if len(query.split()) > 20:  # raised from 15 — Arabic phrases have fewer space-separated tokens
            return None

        lang = language or self._detect_query_language(query)
        prompt_tmpl = self._HYDE_PROMPTS.get(lang, self._HYDE_PROMPTS["en"])
        prompt = prompt_tmpl.format(query=query)

        try:
            import requests
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 120, "temperature": 0.3},
                },
                timeout=8,
            )
            if r.status_code == 200:
                text = r.json().get("response", "").strip()
                if text:
                    logger.debug("HyDE expanded [lang=%s]: %s…", lang, text[:60])
                    return text[:350]
            else:
                logger.debug("HyDE expand: Ollama returned %d", r.status_code)
        except Exception as exc:
            logger.debug("HyDE expansion failed: %s", exc)
        return None


class KeyPointsEmbedder:
    def __init__(self, bge: BGEM3Engine):
        self.bge = bge

    def embed_key_points(self, key_points: List[str]) -> Optional[List[float]]:
        if not key_points:
            return None
        summary = " | ".join(kp.strip() for kp in key_points if kp.strip())
        if not summary:
            return None
        try:
            result = self.bge.encode_batch([summary], with_colbert=False)
            if result and result[0].get("dense"):
                return result[0]["dense"]
        except Exception as exc:
            logger.debug("KeyPoints embed failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# ScoreFusionLayer — RRF with language-adaptive weights
# ══════════════════════════════════════════════════════════════════════

class ScoreFusionLayer:
    """
    Fuse scores from multiple retrieval channels using RRF
    with language-adaptive weights.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[str, float]]],
        language: str = "en",
    ) -> List[Tuple[str, float]]:
        weights = get_lang_weights(language)
        scores: Dict[str, float] = {}

        for channel, results in ranked_lists.items():
            w = weights.get(channel, 0.1)
            for rank, (doc_id, _) in enumerate(results):
                rrf_score = w / (self.k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════════
# EmbeddingGenerator — unified public API
# ══════════════════════════════════════════════════════════════════════

class EmbeddingGenerator:
    """
    Multi-engine embedding generator.
    Always uses BGE-M3. Optionally adds Nomic, ColBERT, HyDE.
    """

    def __init__(
        self,
        bge_engine: Optional[BGEM3Engine] = None,
        batch_size: int = 8,
        enable_nomic: Optional[bool] = None,
        enable_colbert: Optional[bool] = None,
        enable_hyde: Optional[bool] = None,
        enable_cache: Optional[bool] = None,
    ) -> None:
        self.enable_nomic = (
            enable_nomic if enable_nomic is not None
            else _cfg_bool("nomic_enabled", False)
        )
        self.enable_colbert = (
            enable_colbert if enable_colbert is not None
            else _cfg_bool("colbert_enabled", True)
        )
        self.enable_hyde = (
            enable_hyde if enable_hyde is not None
            else _cfg_bool("hyde_enabled", False)
        )
        _enable_cache = (
            enable_cache if enable_cache is not None
            else _cfg_bool("embedding_cache_enabled", True)
        )

        self.bge = bge_engine or BGEM3Engine(
            enable_colbert=self.enable_colbert,
        )
        self.batch_size = batch_size

        # Optional engines — all lazily initialized
        self._nomic: Optional[NomicEngine] = None
        self._nomic_init_done = False        # BUG-1 FIX

        self._expander: Optional[QueryExpander] = None
        self._expander_init_done = False     # BUG-3 FIX

        self._kp_embedder: Optional[KeyPointsEmbedder] = None

        self._cache: Optional[EmbeddingCache] = None
        if _enable_cache:
            self._cache = EmbeddingCache(
                max_size=_cfg_int("embedding_cache_size", 2048),
                ttl_seconds=_cfg_float("embedding_cache_ttl", 86400.0),
            )

        logger.info(
            "EmbeddingGenerator initialized | nomic=%s | colbert=%s | hyde=%s | cache=%s",
            self.enable_nomic, self.enable_colbert, self.enable_hyde,
            "ON" if self._cache else "OFF",
        )

    # ── Lazy init with explicit logging ──────────────────────────────

    def _lazy_init_nomic(self) -> None:
        """
        BUG-1 FIX : init une seule fois, log explicite si disabled.
        """
        if self._nomic_init_done:
            return
        self._nomic_init_done = True

        if not self.enable_nomic:
            logger.debug("Nomic disabled by config (nomic_enabled=False)")
            return

        self._nomic = NomicEngine(
            truncate_dim=_cfg_int("nomic_truncate_dim", 256),
        )
        self._nomic.load()

        if self._nomic.available:
            logger.info("✅ Nomic engine ready (truncate=%dd)", self._nomic.truncate_dim)
        else:
            logger.warning(
                "⚠️  Nomic engine NOT available: %s. "
                "All Nomic vectors will be None — falling back to BGE-only.",
                self._nomic._load_error or "unknown reason",
            )

    def _lazy_init_hyde(self) -> None:
        """
        BUG-3 FIX : init une seule fois, pas de check bloquant dans __init__.
        """
        if self._expander_init_done:
            return
        self._expander_init_done = True

        if not self.enable_hyde:
            logger.debug("HyDE disabled by config (hyde_enabled=False)")
            return

        self._expander = QueryExpander()
        # Note: Ollama check is lazy inside QueryExpander.available

    def _lazy_init_kp(self) -> None:
        if self._kp_embedder is None:
            self._kp_embedder = KeyPointsEmbedder(self.bge)

    # ── Query embedding ──────────────────────────────────────────────

    def embed_query(self, text: str, language: Optional[str] = None) -> QueryEmbedding:
        """Encode a search query → QueryEmbedding."""
        text = _normalize_text(text)
        if not text:
            raise ValueError("Cannot embed empty query.")

        # HyDE expansion
        expanded = None
        self._lazy_init_hyde()
        if self._expander and self._expander.available:
            expanded = self._expander.expand(text, language=language)
            if expanded:
                logger.debug("HyDE expanded query (+%d chars)", len(expanded))

        # BGE-M3
        embed_text = f"{text} {expanded}" if expanded else text
        bge_result = self.bge.encode_query(embed_text, with_colbert=self.enable_colbert)

        bge_dense = bge_result.get("dense", [])
        bge_sparse = self.bge.sparse_to_qdrant(bge_result.get("sparse"))
        colbert_query = bge_result.get("colbert")

        # Nomic
        nomic_dense = None
        self._lazy_init_nomic()
        if self._nomic and self._nomic.available:
            nomic_dense = self._nomic.encode_query(text)

        return QueryEmbedding(
            text=text,
            bge_dense=bge_dense,
            bge_sparse=bge_sparse,
            nomic_dense=nomic_dense,
            colbert_query=colbert_query,
            expanded_text=expanded,
        )

    # Alias
    def encode_query(self, text: str, language: Optional[str] = None) -> QueryEmbedding:
        return self.embed_query(text, language=language)

    def generate_query(self, text: str) -> QueryEmbedding:
        return self.embed_query(text, language=language)

    # ── Chunk embedding ──────────────────────────────────────────────

    def generate_chunks(
        self,
        semantic_chunks: Sequence[Any],
        video_id: str = "",
    ) -> List[EmbeddedChunk]:
        """
        Encode a batch of semantic chunks → List[EmbeddedChunk].
        Primary public API called by api/main.py.
        """
        if not semantic_chunks:
            return []
        video_id = str(video_id or "unknown_video").strip()

        prepared = [self._extract_payload(c, video_id) for c in semantic_chunks]
        prepared = [p for p in prepared if p["text"]]
        if not prepared:
            return []

        texts = [p["search_text"] or p["text"] for p in prepared]
        total = len(prepared)
        t0 = time.time()

        # ── Check cache ──────────────────────────────────────────────
        cached_indices: Dict[int, Dict[str, Any]] = {}
        uncached_indices: List[int] = []

        if self._cache:
            for i, txt in enumerate(texts):
                hit = self._cache.get(txt)
                if hit:
                    cached_indices[i] = hit
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(total))

        if cached_indices:
            logger.info(
                "EmbeddingCache: %d hits, %d to encode",
                len(cached_indices), len(uncached_indices),
            )

        # ── Encode uncached with BGE-M3 ──────────────────────────────
        uncached_texts = [texts[i] for i in uncached_indices]
        bge_results: List[Dict[str, Any]] = []
        if uncached_texts:
            bge_results = self.bge.encode_batch(
                uncached_texts, with_colbert=self.enable_colbert,
            )

        # ── Nomic for uncached ────────────────────────────────────────
        nomic_results: List[Optional[List[float]]] = [None] * len(uncached_texts)
        self._lazy_init_nomic()
        if self._nomic and self._nomic.available and uncached_texts:
            try:
                nomic_vecs = self._nomic.encode_documents(uncached_texts)
                for i, vec in enumerate(nomic_vecs):
                    if i < len(nomic_results):
                        nomic_results[i] = vec
            except Exception as exc:
                logger.warning("Nomic batch encode failed: %s", exc)

        # ── Build encoding map ────────────────────────────────────────
        encoding_map: Dict[int, Dict[str, Any]] = {}
        for rel_idx, abs_idx in enumerate(uncached_indices):
            if rel_idx < len(bge_results):
                entry = {
                    "dense": bge_results[rel_idx].get("dense"),
                    "sparse": bge_results[rel_idx].get("sparse"),
                    "colbert": bge_results[rel_idx].get("colbert"),
                    "nomic": (
                        nomic_results[rel_idx]
                        if rel_idx < len(nomic_results) else None
                    ),
                }
                encoding_map[abs_idx] = entry
                if self._cache:
                    self._cache.put(texts[abs_idx], entry)

        # ── Merge cached + uncached → EmbeddedChunk ───────────────────
        self._lazy_init_kp()
        results: List[EmbeddedChunk] = []

        for i, item in enumerate(prepared):
            if i in cached_indices:
                enc = cached_indices[i]
            elif i in encoding_map:
                enc = encoding_map[i]
            else:
                continue

            dense = enc.get("dense")
            if not dense:
                continue

            # Key points embedding
            kp_vec = None
            if self._kp_embedder and item.get("key_points"):
                kp_vec = self._kp_embedder.embed_key_points(item["key_points"])

            results.append(EmbeddedChunk(
                chunk_id=item["chunk_id"],
                video_id=item["video_id"],
                text=item["text"],
                original_text=item["original_text"],
                search_text=item["search_text"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                language=item["language"],
                speaker=item["speaker"],
                topic=item["topic"],
                chunk_type=item["chunk_type"],
                bge_dense=dense,
                bge_sparse=self.bge.sparse_to_qdrant(enc.get("sparse")),
                nomic_dense=enc.get("nomic"),
                colbert_vecs=enc.get("colbert"),
                key_points_vec=kp_vec,
                key_points=item.get("key_points", []),
                metadata=item["metadata"],
            ))

        elapsed = time.time() - t0
        cache_info = f" | cache={self._cache.stats}" if self._cache else ""
        logger.info(
            "Embedding complete | video=%s | %d/%d chunks in %.2fs%s",
            video_id, len(results), total, elapsed, cache_info,
        )

        # Persist cache after batch
        if self._cache:
            self._cache.save()

        return results

    # Aliases for api/main.py _call() compatibility
    def embed(self, chunks: Sequence[Any], video_id: str = "") -> List[EmbeddedChunk]:
        return self.generate_chunks(chunks, video_id=video_id)

    def embed_chunks(self, chunks: Sequence[Any], video_id: str = "") -> List[EmbeddedChunk]:
        return self.generate_chunks(chunks, video_id=video_id)

    def process(self, chunks: Sequence[Any], video_id: str = "") -> List[EmbeddedChunk]:
        return self.generate_chunks(chunks, video_id=video_id)

    # ── Payload extraction ────────────────────────────────────────────

    def _extract_payload(self, chunk: Any, video_id: str) -> Dict[str, Any]:
        text = _normalize_text(_safe_attr(chunk, "text", "content", default=""))
        original_text = (
            _normalize_text(_safe_attr(chunk, "original_text", default="")) or text
        )
        search_text = (
            _normalize_text(_safe_attr(chunk, "search_text", default="")) or text
        )
        start_time = _to_float(_safe_attr(chunk, "start_time", "start", default=0.0))
        end_time = _to_float(_safe_attr(chunk, "end_time", "end", default=0.0))
        language = (
            _normalize_text(_safe_attr(chunk, "language", default="unknown"))
            or "unknown"
        )
        speaker = (
            _normalize_text(_safe_attr(chunk, "speaker", default="unknown"))
            or "unknown"
        )
        topic = (
            _normalize_text(_safe_attr(chunk, "topic", default="general"))
            or "general"
        )
        chunk_type = (
            _normalize_text(_safe_attr(chunk, "chunk_type", default="speech"))
            or "speech"
        )
        chunk_id = _safe_attr(chunk, "chunk_id", "id", default=None)
        if not chunk_id:
            chunk_id = _deterministic_id(video_id, text, start_time, end_time)

        metadata = _safe_attr(chunk, "metadata", default={}) or {}
        metadata = dict(metadata) if isinstance(metadata, dict) else {}

        # Preserve the original CLIP embedding for visual retrieval.
        # Without this, /vision/ask cannot recover frame vectors after
        # the BGE embedding stage stores the chunk in VideoStore.
        clip_embedding = _safe_attr(chunk, "clip_embedding", "embedding", default=None)
        clip_embedding_dim = _safe_attr(chunk, "clip_embedding_dim", "embedding_dim", default=0)
        if clip_embedding is not None and "clip_embedding" not in metadata:
            metadata["clip_embedding"] = clip_embedding
        if clip_embedding_dim and "clip_embedding_dim" not in metadata:
            try:
                metadata["clip_embedding_dim"] = int(clip_embedding_dim)
            except Exception:
                metadata["clip_embedding_dim"] = 0

        key_points = _safe_attr(chunk, "key_points", default=[]) or []
        if isinstance(key_points, str):
            key_points = [key_points]

        return {
            "chunk_id": str(chunk_id),
            "video_id": str(video_id),
            "text": text,
            "original_text": original_text,
            "search_text": search_text,
            "start_time": start_time,
            "end_time": end_time,
            "language": language,
            "speaker": speaker,
            "topic": topic,
            "chunk_type": chunk_type,
            "key_points": key_points,
            "metadata": metadata,
        }