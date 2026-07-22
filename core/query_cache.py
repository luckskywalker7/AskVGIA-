"""
core/query_cache.py — Semantic Query Cache v1.0
================================================

Stores (video_id, question → answer) pairs in a SQLite database so that
identical or semantically-similar questions asked about the same video are
answered instantly from cache without re-running the full RAG pipeline.

How it works
------------
1. On every /ask call we compute a dense embedding of the question.
2. We look up all cached entries for that video_id and measure cosine
   similarity between the incoming question embedding and every stored
   question embedding.
3. If the best match exceeds `similarity_threshold` (default 0.92) we
   return the cached answer immediately — no search, no LLM call.
4. Otherwise we run the normal pipeline and save the result to the cache.

Cache key   : (video_id, question_embedding)
Cache value : full /ask response dict (answer, citations, llm, results …)

Storage
-------
• SQLite file at  <storage_dir>/query_cache.db
• Table `query_cache`:
    id              INTEGER PRIMARY KEY
    video_id        TEXT
    question        TEXT        ← original question text
    question_emb    BLOB        ← float32 numpy array (dense embedding)
    response_json   TEXT        ← JSON-serialised /ask response
    hit_count       INTEGER     ← how many times this entry was reused
    created_at      REAL        ← unix timestamp
    last_hit_at     REAL        ← unix timestamp of last cache hit

Configuration (all optional, via settings)
-------------------------------------------
    query_cache_enabled          bool    = True
    similarity_threshold: float = 0.85   (0-1, higher = stricter — lowered from 0.92 for better hit rate)
    query_cache_llm_fallback_threshold  float   = 0.85  (disabled; kept for config compat)
    query_cache_max_entries      int     = 10_000
    query_cache_ttl_days         float   = 30.0   (0 = never expire)

Public API
----------
    cache = QueryCache(storage_dir="data/faiss_store", embedder=embedder)

    # Try cache before heavy search
    hit = cache.lookup(question="What is the main topic?", video_id="abc")
    if hit:
        return hit   # ← instant response

    # ... run normal pipeline ...

    # Save to cache after generating a real answer
    cache.save(
        question="What is the main topic?",
        video_id="abc",
        response=ask_response_dict,
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── optional numpy / scipy ────────────────────────────────────────────
try:
    import numpy as np  # type: ignore
    _NP_OK = True
except ImportError:
    np = None  # type: ignore
    _NP_OK = False


# ── LLM equivalence check (lazy import to avoid circular deps) ───────
def _llm_questions_equivalent(q1: str, q2: str) -> bool:
    """
    Disabled: LLM fallback for equivalence check adds 15-60s latency,
    defeating the purpose of a fast cache. Cosine similarity alone is
    sufficient — just tune similarity_threshold instead.
    """
    return False


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity between two 1-D float32 arrays."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _emb_to_blob(emb: "np.ndarray") -> bytes:
    """Serialise a float32 numpy array to raw bytes."""
    arr = np.asarray(emb, dtype="float32").flatten()
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _blob_to_emb(blob: bytes) -> "np.ndarray":
    """Deserialise raw bytes back to a float32 numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype="float32")


def _cfg(name: str, default: Any) -> Any:
    try:
        from config.settings import settings  # type: ignore
        return getattr(settings, name, default)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────
# QueryCache
# ─────────────────────────────────────────────────────────────────────

class QueryCache:
    """
    Semantic cache for /ask query → response pairs.

    Thread-safe: uses SQLite WAL mode + a per-instance threading.Lock.
    """

    def __init__(
        self,
        storage_dir: str = "data/faiss_store",
        embedder: Any = None,
        similarity_threshold: Optional[float] = None,
        max_entries: Optional[int] = None,
        ttl_days: Optional[float] = None,
        enabled: Optional[bool] = None,
        llm_fallback_threshold: Optional[float] = None,
    ):
        self.enabled: bool = (
            enabled
            if enabled is not None
            else bool(_cfg("query_cache_enabled", True))
        )
        self.similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else float(_cfg("query_cache_similarity", 0.85))
        )
        # Cosine floor below which we skip the entry entirely (no LLM check).
        # Between llm_fallback_threshold and similarity_threshold → cache MISS
        # (LLM fallback has been disabled — too slow for real-time use).
        self.llm_fallback_threshold: float = (
            llm_fallback_threshold
            if llm_fallback_threshold is not None
            else float(_cfg("query_cache_llm_fallback_threshold", 0.85))
        )
        self.max_entries: int = (
            max_entries
            if max_entries is not None
            else int(_cfg("query_cache_max_entries", 10_000))
        )
        self.ttl_seconds: float = (
            (ttl_days if ttl_days is not None
             else float(_cfg("query_cache_ttl_days", 30.0))) * 86_400
        )
        self._embedder = embedder

        # DB path
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(storage_dir) / "query_cache.db")
        self._init_db()

        if not self.enabled:
            logger.info("QueryCache: disabled (query_cache_enabled=False)")
        else:
            n = self._count()
            logger.info(
                "QueryCache ready | db=%s | entries=%d | "
                "threshold=%.2f | llm_fallback=%.2f | ttl_days=%.0f",
                self._db_path, n,
                self.similarity_threshold,
                self.llm_fallback_threshold,
                self.ttl_seconds / 86_400,
            )

    # ─────────────────────────────────────────────────────────────────
    # DB init
    # ─────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id        TEXT    NOT NULL,
                    question        TEXT    NOT NULL,
                    question_emb    BLOB,
                    response_json   TEXT    NOT NULL,
                    hit_count       INTEGER NOT NULL DEFAULT 0,
                    created_at      REAL    NOT NULL,
                    last_hit_at     REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qc_video "
                "ON query_cache(video_id)"
            )
            conn.commit()

    # ─────────────────────────────────────────────────────────────────
    # Public: lookup
    # ─────────────────────────────────────────────────────────────────

    def lookup(
        self,
        question: str,
        video_id: Optional[str],
        embedder: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find a cached response for this (video_id, question) pair.

        Returns the cached response dict if a match is found,
        or None if no suitable cache entry exists.

        Parameters
        ----------
        question  : the raw question string
        video_id  : the video being queried (None = global)
        embedder  : optional override for the instance embedder
        """
        if not self.enabled:
            return None
        if not _NP_OK:
            return self._lookup_exact(question, video_id)

        emb = self._embed(question, embedder)
        if emb is None:
            # Fall back to exact-text match
            return self._lookup_exact(question, video_id)

        return self._lookup_semantic(question, video_id, emb)

    # ─────────────────────────────────────────────────────────────────
    # Public: save
    # ─────────────────────────────────────────────────────────────────

    def save(
        self,
        question: str,
        video_id: Optional[str],
        response: Dict[str, Any],
        embedder: Any = None,
    ) -> None:
        """
        Persist a (video_id, question → response) entry to the cache.

        If an entry with a very similar question already exists for
        this video (above threshold), we update it instead of inserting.
        """
        if not self.enabled:
            return

        emb = self._embed(question, embedder) if _NP_OK else None
        emb_blob = _emb_to_blob(emb) if emb is not None else None

        try:
            response_json = json.dumps(response, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("QueryCache.save: JSON serialisation failed: %s", e)
            return

        now = time.time()

        # Check if we should UPDATE an existing near-identical entry
        existing_id = self._find_similar_id(question, video_id, emb)
        with self._connect() as conn:
            if existing_id is not None:
                conn.execute(
                    "UPDATE query_cache "
                    "SET response_json=?, last_hit_at=?, question_emb=? "
                    "WHERE id=?",
                    (response_json, now, emb_blob, existing_id),
                )
            else:
                conn.execute(
                    "INSERT INTO query_cache "
                    "(video_id, question, question_emb, response_json, "
                    " hit_count, created_at, last_hit_at) "
                    "VALUES (?, ?, ?, ?, 0, ?, ?)",
                    (video_id or "", question, emb_blob,
                     response_json, now, now),
                )
            conn.commit()

        # Evict oldest entries if over the limit
        self._evict_if_needed()
        logger.debug(
            "QueryCache saved | video_id=%s | q=%r | existing_updated=%s",
            video_id, question[:60], existing_id is not None,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public: invalidate a video (call on /delete)
    # ─────────────────────────────────────────────────────────────────

    def delete_video(self, video_id: str) -> int:
        """Remove all cached entries for a given video_id."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM query_cache WHERE video_id=?", (video_id,)
            )
            conn.commit()
            n = cur.rowcount
        logger.info("QueryCache: deleted %d entries for video_id=%s", n, video_id)
        return n

    # ─────────────────────────────────────────────────────────────────
    # Public: stats
    # ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM query_cache"
            ).fetchone()[0]
            videos = conn.execute(
                "SELECT COUNT(DISTINCT video_id) FROM query_cache"
            ).fetchone()[0]
            total_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM query_cache"
            ).fetchone()[0]
            top = conn.execute(
                "SELECT video_id, question, hit_count "
                "FROM query_cache ORDER BY hit_count DESC LIMIT 5"
            ).fetchall()
        return {
            "enabled": self.enabled,
            "total_entries": total,
            "total_videos": videos,
            "total_hits": total_hits,
            "similarity_threshold": self.similarity_threshold,
            "top_entries": [
                {"video_id": r["video_id"], "question": r["question"],
                 "hit_count": r["hit_count"]}
                for r in top
            ],
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _embed(self, text: str, embedder: Any = None) -> Optional["np.ndarray"]:
        """Embed `text` using the provided or stored embedder."""
        emb_obj = embedder or self._embedder
        if emb_obj is None:
            return None
        try:
            result = emb_obj.embed_query(text)
            # EmbeddingGenerator returns a QueryEmbedding dataclass
            if hasattr(result, "bge_dense") and result.bge_dense is not None:
                return np.asarray(result.bge_dense, dtype="float32")
            # Plain numpy array / list
            if isinstance(result, (list, np.ndarray)):
                return np.asarray(result, dtype="float32")
        except Exception as e:
            logger.debug("QueryCache._embed failed: %s", e)
        return None

    def _rows_for_video(
        self, video_id: Optional[str],
    ) -> List[sqlite3.Row]:
        """Return all non-expired rows for this video_id."""
        vid = video_id or ""
        now = time.time()
        with self._connect() as conn:
            if self.ttl_seconds > 0:
                cutoff = now - self.ttl_seconds
                rows = conn.execute(
                    "SELECT id, question, question_emb, response_json, hit_count "
                    "FROM query_cache "
                    "WHERE video_id=? AND created_at >= ?",
                    (vid, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, question, question_emb, response_json, hit_count "
                    "FROM query_cache WHERE video_id=?",
                    (vid,),
                ).fetchall()
        return rows

    def _lookup_semantic(
        self,
        question: str,
        video_id: Optional[str],
        emb: "np.ndarray",
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best semantically-similar cached entry.

        Decision logic:
          score >= similarity_threshold        → immediate cache HIT
          llm_fallback_threshold <= score
                              < similarity_threshold → ask LLM to decide
          score < llm_fallback_threshold       → cache MISS (too different)
        """
        rows = self._rows_for_video(video_id)
        if not rows:
            return None

        best_id = None
        best_score = -1.0
        best_response_json = None
        best_hit_count = 0
        best_question = ""

        for row in rows:
            blob = row["question_emb"]
            if blob is None:
                # Exact fallback for this row
                if row["question"].strip().lower() == question.strip().lower():
                    best_id = row["id"]
                    best_score = 1.0
                    best_response_json = row["response_json"]
                    best_hit_count = row["hit_count"]
                    best_question = row["question"]
                continue

            cached_emb = _blob_to_emb(blob)
            score = _cosine(emb, cached_emb)
            if score > best_score:
                best_score = score
                best_id = row["id"]
                best_response_json = row["response_json"]
                best_hit_count = row["hit_count"]
                best_question = row["question"]

        # ── Hard miss: too dissimilar even for LLM check ─────────────
        if best_score < self.llm_fallback_threshold or best_id is None:
            logger.debug(
                "QueryCache MISS (below floor) | video=%s | q=%r | best_score=%.3f",
                video_id, question[:60], best_score,
            )
            return None

        # ── Direct hit: above cosine threshold ───────────────────────
        if best_score >= self.similarity_threshold:
            return self._build_hit(
                best_id, best_response_json, best_hit_count,
                best_score, question, video_id, method="cosine",
            )

        # ── Grey zone: llm_fallback_threshold <= score < similarity_threshold
        # Ask the LLM whether the two questions are equivalent.
        logger.debug(
            "QueryCache grey zone | video=%s | score=%.3f | asking LLM | "
            "cached_q=%r | new_q=%r",
            video_id, best_score, best_question[:60], question[:60],
        )
        if _llm_questions_equivalent(question, best_question):
            logger.info(
                "QueryCache HIT (LLM-confirmed) ✓ | video=%s | score=%.3f | "
                "q=%r ≈ cached=%r",
                video_id, best_score, question[:60], best_question[:60],
            )
            return self._build_hit(
                best_id, best_response_json, best_hit_count,
                best_score, question, video_id, method="llm",
            )

        logger.debug(
            "QueryCache MISS (LLM rejected) | video=%s | score=%.3f | q=%r",
            video_id, best_score, question[:60],
        )
        return None

    def _build_hit(
        self,
        row_id: int,
        response_json: str,
        hit_count: int,
        score: float,
        question: str,
        video_id: Optional[str],
        method: str = "cosine",
    ) -> Optional[Dict[str, Any]]:
        """Update DB stats and return the annotated cached response."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE query_cache "
                "SET hit_count=hit_count+1, last_hit_at=? WHERE id=?",
                (now, row_id),
            )
            conn.commit()

        try:
            response = json.loads(response_json)
        except Exception:
            return None

        response["_cache_hit"] = True
        response["_cache_similarity"] = round(score, 4)
        response["_cache_hit_count"] = hit_count + 1
        response["_cache_method"] = method   # "cosine" | "llm"

        logger.info(
            "QueryCache HIT ✓ | method=%s | video=%s | q=%r | score=%.3f | hits=%d",
            method, video_id, question[:60], score, hit_count + 1,
        )
        return response

    def _lookup_exact(
        self,
        question: str,
        video_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Pure exact-text match fallback (used when numpy is unavailable)."""
        vid = video_id or ""
        now = time.time()
        with self._connect() as conn:
            if self.ttl_seconds > 0:
                row = conn.execute(
                    "SELECT id, response_json, hit_count FROM query_cache "
                    "WHERE video_id=? AND question=? AND created_at >= ?",
                    (vid, question, now - self.ttl_seconds),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id, response_json, hit_count FROM query_cache "
                    "WHERE video_id=? AND question=?",
                    (vid, question),
                ).fetchone()

            if row is None:
                return None

            conn.execute(
                "UPDATE query_cache "
                "SET hit_count=hit_count+1, last_hit_at=? WHERE id=?",
                (now, row["id"]),
            )
            conn.commit()

        try:
            response = json.loads(row["response_json"])
        except Exception:
            return None

        response["_cache_hit"] = True
        response["_cache_similarity"] = 1.0
        response["_cache_hit_count"] = row["hit_count"] + 1
        logger.info(
            "QueryCache HIT (exact) ✓ | video=%s | q=%r | hits=%d",
            video_id, question[:60], row["hit_count"] + 1,
        )
        return response

    def _find_similar_id(
        self,
        question: str,
        video_id: Optional[str],
        emb: Optional["np.ndarray"],
    ) -> Optional[int]:
        """
        Return the DB id of an existing entry that matches this question
        semantically (or exactly if emb is None), or None.
        Used by save() to UPDATE instead of INSERT duplicates.
        """
        if emb is None:
            # Exact match only
            vid = video_id or ""
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id FROM query_cache WHERE video_id=? AND question=?",
                    (vid, question),
                ).fetchone()
            return row["id"] if row else None

        rows = self._rows_for_video(video_id)
        best_id = None
        best_score = -1.0
        for row in rows:
            if row["question_emb"] is None:
                continue
            score = _cosine(emb, _blob_to_emb(row["question_emb"]))
            if score > best_score:
                best_score = score
                best_id = row["id"]

        if best_score >= self.similarity_threshold:
            return best_id
        return None

    def _count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM query_cache"
            ).fetchone()[0]

    def _evict_if_needed(self) -> None:
        """Delete oldest entries once we exceed max_entries."""
        if self.max_entries <= 0:
            return
        total = self._count()
        if total <= self.max_entries:
            return
        excess = total - self.max_entries
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM query_cache WHERE id IN ("
                "  SELECT id FROM query_cache "
                "  ORDER BY last_hit_at ASC LIMIT ?"
                ")",
                (excess,),
            )
            conn.commit()
        logger.info("QueryCache evicted %d old entries", excess)


__all__ = ["QueryCache"]