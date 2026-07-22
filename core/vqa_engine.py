"""
core/vqa_engine.py
==================

Visual Question Answering engine.

BUG-2 FIX: Eliminated the duplicate InMemoryFrameIndex with its own
FAISS + snapshot files.  VQAEngine now queries storage.py's VideoStore
(the `visual_clip` collection / chunk_type="visual" records) directly.

One index, one source of truth:
  - Ingestion  →  vision_indexer.py → store.add_embedded_chunks(chunks)
  - Retrieval  →  VQAEngine asks store for visual chunks by CLIP vector

The old InMemoryFrameIndex class is kept as a thin compatibility shim
(raises NotImplementedError on construction so callers discover the
change immediately rather than silently desync'ing).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .vlm_contracts import VQAResult
from .vlm_client import VLMClient
from .clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("NumPy is required for vqa_engine.py") from exc


# ---------------------------------------------------------------------------
# Compatibility shim — emits a clear error so callers migrate fast
# ---------------------------------------------------------------------------

class InMemoryFrameIndex:
    """
    REMOVED — BUG-2 FIX.

    This class used its own FAISS + snapshot files that were separate from
    VideoStore, causing de-sync on restart and after partial ingestion
    failures.

    Migration path:
        1. Remove all `InMemoryFrameIndex(...)` constructions.
        2. Pass the shared `VideoStore` instance to `VQAEngine` via the
           `store` parameter.
        3. Visual chunks are already in the store after
           `store.add_embedded_chunks(visual_chunks)` in main.py.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "InMemoryFrameIndex has been removed (BUG-2 FIX). "
            "Pass the shared VideoStore instance to VQAEngine(store=...) instead. "
            "See the class docstring for the migration path."
        )


# ---------------------------------------------------------------------------
# VisualStoreAdapter
# ---------------------------------------------------------------------------

class VisualStoreAdapter:
    """
    Adapts VideoStore + VisionIndexer for visual-frame retrieval.

    Search priority:
    1. VisionIndexer._clip_faiss (dedicated CLIP FAISS — O(log n) ANN)
    2. _meta linear scan fallback (O(n) — only if CLIP FAISS unavailable)
    """

    def __init__(self, store: Any, vision_indexer: Any = None) -> None:
        self._store = store
        self._vision = vision_indexer  # VisionIndexer instance for CLIP FAISS

    def search_by_vector(
        self,
        vector,
        *,
        top_k: int = 10,
        video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Path 1 — dedicated CLIP FAISS index (fast, accurate)
        if self._vision is not None:
            clip_faiss = getattr(self._vision, "_clip_faiss", None)
            if clip_faiss is not None and clip_faiss.ntotal > 0:
                raw = self._vision.search_by_image_embedding(
                    list(vector), top_k=top_k, video_id=video_id
                )
                if raw:
                    return self._enrich_from_store(raw)

        # Path 2 — linear scan fallback
        return self._linear_scan(vector, top_k=top_k, video_id=video_id)

    def _enrich_from_store(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Add text metadata from _meta to CLIP FAISS search results."""
        meta_store = getattr(self._store, "_meta", {})
        enriched = []
        for r in results:
            chunk_id = r.get("chunk_id", "")
            meta = meta_store.get(chunk_id, {})
            chunk_meta = meta.get("metadata", {}) or {}
            enriched.append({
                "chunk_id":   chunk_id,
                "video_id":   r.get("video_id", ""),
                "frame_path": r.get("frame_path", ""),
                "timestamp":  float(r.get("timestamp") or 0.0),
                "score":      float(r.get("score", 0.0)),
                "caption":    chunk_meta.get("caption") or chunk_meta.get("short_caption") or "",
                "board_text": chunk_meta.get("board_text", "") or "",
                "screen_text": chunk_meta.get("screen_text", "") or "",
                "all_visible_text": chunk_meta.get("all_visible_text", "") or "",
                "tags":       chunk_meta.get("tags", []) or [],
                "detected_objects": chunk_meta.get("detected_objects", []) or [],
                "spatial_layout": chunk_meta.get("spatial_layout", "") or "",
                "object_counts": chunk_meta.get("object_counts", {}) or {},
                "ocr_text":   chunk_meta.get("ocr_text", "") or "",
                "chunk_type": "visual",
                "metadata":   chunk_meta,
            })
        return enriched

    def _linear_scan(
        self, vector, *, top_k: int, video_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        visual_meta = self._collect_visual_metadata(video_id=video_id)
        if not visual_meta:
            return []

        qvec = np.asarray(vector, dtype=np.float32).flatten()
        qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for meta in visual_meta:
            emb = meta.get("_clip_embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32).flatten()
            if v.size != qnorm.size:
                continue
            vn = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(qnorm, vn))
            scored.append((score, meta))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, meta in scored[:top_k]:
            chunk_meta = meta.get("metadata", {}) or {}
            results.append({
                "chunk_id":   meta.get("chunk_id", ""),
                "video_id":   meta.get("video_id", ""),
                "frame_path": chunk_meta.get("frame_path", ""),
                "timestamp":  float(meta.get("start_time") or 0.0),
                "score":      score,
                "caption":    chunk_meta.get("caption") or chunk_meta.get("short_caption") or "",
                "board_text": chunk_meta.get("board_text", "") or "",
                "screen_text": chunk_meta.get("screen_text", "") or "",
                "all_visible_text": chunk_meta.get("all_visible_text", "") or "",
                "tags":       chunk_meta.get("tags", []) or [],
                "detected_objects": chunk_meta.get("detected_objects", []) or [],
                "spatial_layout": chunk_meta.get("spatial_layout", "") or "",
                "object_counts": chunk_meta.get("object_counts", {}) or {},
                "ocr_text":   chunk_meta.get("ocr_text", "") or "",
                "chunk_type": "visual",
                "metadata":   chunk_meta,
            })
        return results

    def _collect_visual_metadata(
        self, *, video_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        store = self._store
        meta_store: Dict[str, Dict[str, Any]] = getattr(store, "_meta", {})
        video_chunks: Dict[str, Any] = getattr(store, "_video_chunks", {})

        if video_id:
            chunk_ids = video_chunks.get(video_id, set())
            candidates = {
                cid: meta_store[cid]
                for cid in chunk_ids
                if cid in meta_store
            }
        else:
            candidates = meta_store

        visual: List[Dict[str, Any]] = []
        for cid, meta in candidates.items():
            if meta.get("chunk_type") != "visual":
                continue
            chunk_meta = meta.get("metadata", {}) or {}
            clip_emb = (
                chunk_meta.get("clip_embedding")
                or meta.get("clip_embedding")
                or chunk_meta.get("embedding")
                or meta.get("embedding")
            )
            row = dict(meta)
            row["_clip_embedding"] = clip_emb
            visual.append(row)

        return visual

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search_by_vector(
        self,
        vector,
        *,
        top_k: int = 10,
        video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find visual chunks most similar to `vector` (CLIP embedding).

        Strategy:
          1. Pull all visual-chunk metadata from the store (filtered by
             video_id when provided).
          2. Cosine-rank against `vector` using the stored CLIP embedding.
          3. Return the top_k results in the same dict shape the old
             InMemoryFrameIndex returned so VQAEngine needs zero changes.
        """
        visual_meta = self._collect_visual_metadata(video_id=video_id)
        if not visual_meta:
            return []

        qvec = np.asarray(vector, dtype=np.float32).flatten()
        qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for meta in visual_meta:
            emb = meta.get("_clip_embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32).flatten()
            if v.size != qnorm.size:
                continue
            vn = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(qnorm, vn))
            scored.append((score, meta))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, meta in scored[:top_k]:
            chunk_meta = meta.get("metadata", {}) or {}
            results.append({
                "chunk_id":   meta.get("chunk_id", ""),
                "video_id":   meta.get("video_id", ""),
                "frame_path": chunk_meta.get("frame_path", ""),
                "timestamp":  float(meta.get("start_time") or 0.0),
                "score":      score,
                "caption":    chunk_meta.get("caption") or chunk_meta.get("short_caption") or "",
                "tags":       chunk_meta.get("tags", []) or [],
                "detected_objects": chunk_meta.get("detected_objects", []) or [],
                "spatial_layout": chunk_meta.get("spatial_layout", "") or "",
                "object_counts": chunk_meta.get("object_counts", {}) or {},
                "ocr_text":   chunk_meta.get("ocr_text", "") or "",
                "chunk_type": "visual",
                "metadata":   chunk_meta,
            })
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_visual_metadata(
        self, *, video_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Walk VideoStore._meta to collect all visual chunks.

        We access the private `_meta` dict directly because VideoStore
        does not expose a "list chunks by type" public API.  This is an
        internal coupling that is acceptable given both classes live in
        the same package.  If VideoStore later gains
        `list_chunks(chunk_type=...)`, replace this with that call.
        """
        store = self._store
        meta_store: Dict[str, Dict[str, Any]] = getattr(store, "_meta", {})
        video_chunks: Dict[str, Any] = getattr(store, "_video_chunks", {})

        if video_id:
            chunk_ids = video_chunks.get(video_id, set())
            candidates = {
                cid: meta_store[cid]
                for cid in chunk_ids
                if cid in meta_store
            }
        else:
            candidates = meta_store

        visual: List[Dict[str, Any]] = []
        for cid, meta in candidates.items():
            if meta.get("chunk_type") != "visual":
                continue
            # CLIP embedding persistence across the pipeline:
            #   vision_indexer.VisualChunk.embedding
            #      -> embeddings._extract_payload(...)
            #      -> metadata["clip_embedding"]
            #      -> storage._meta[chunk_id]["clip_embedding"]
            # Keep backward-compatible fallbacks for older snapshots.
            chunk_meta = meta.get("metadata", {}) or {}
            clip_emb = (
                chunk_meta.get("clip_embedding")
                or meta.get("clip_embedding")
                or chunk_meta.get("embedding")
                or meta.get("embedding")
            )
            row = dict(meta)
            row["_clip_embedding"] = clip_emb  # may be None — filtered later
            visual.append(row)

        return visual


# ---------------------------------------------------------------------------
# FrameIndex protocol (unchanged — VQAEngine depends on this)
# ---------------------------------------------------------------------------

class FrameIndex(Protocol):
    def search_by_vector(
        self,
        vector,
        *,
        top_k: int = 10,
        video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# VQAEngine
# ---------------------------------------------------------------------------

class VQAEngine:
    """
    Orchestrates: text query → CLIP embedding → frame retrieval → VLM VQA.

    BUG-2 FIX: `frame_index` is now a VisualStoreAdapter (or any FrameIndex
    implementor) that wraps the *shared* VideoStore.  There is no longer a
    separate InMemoryFrameIndex with its own FAISS files.

    Typical construction in main.py:

        from core.vqa_engine import VQAEngine, VisualStoreAdapter

        vqa_engine = VQAEngine(
            vlm=vlm_client,
            clip=clip_embedder,
            frame_index=VisualStoreAdapter(store),   # ← shared VideoStore
        )
    """

    def __init__(
        self,
        *,
        vlm: VLMClient,
        clip: CLIPEmbedder,
        frame_index: FrameIndex,
        top_k_retrieve: int = 12,
        top_n_answer: int = 4,
    ):
        self.vlm = vlm
        self.clip = clip
        self.frame_index = frame_index
        self.top_k_retrieve = int(top_k_retrieve)
        self.top_n_answer = int(top_n_answer)

    def _normalized_score(self, raw_score: Any, chunk_type: str = "visual") -> float:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0
        if chunk_type == "visual" and -1.0 <= score <= 1.0:
            return max(0.0, min(1.0, (score + 1.0) / 2.0))
        if 0.0 <= score <= 1.0:
            return score
        return 1.0 / (1.0 + math.exp(-score))

    def _adaptive_answer_count(self, candidates: List[Dict[str, Any]], requested_top_n: Optional[int]) -> int:
        if not candidates:
            return 0
        if requested_top_n is not None:
            return max(1, min(int(requested_top_n), len(candidates)))

        normalized = [
            self._normalized_score(c.get("score"), c.get("chunk_type", "visual"))
            for c in candidates
        ]
        if not normalized:
            return 0

        min_keep = min(4, len(normalized))
        max_keep = min(len(normalized), max(6, self.top_n_answer + 2))
        keep = min_keep
        top_score = normalized[0]
        # CLIP text<->image cosine is compressed: a CORRECT match is often only
        # 0.20-0.28 raw (normalizes to ~0.60-0.64). The old floor of
        # max(0.42, top*0.72) discarded correct-but-lower frames, producing the
        # "no relevant frame found" symptom. Use a much softer relative floor and
        # a low absolute floor so weak-but-plausible frames still reach the VLM,
        # which makes the final visible/not-visible call.
        floor = max(0.30, top_score * 0.55)

        for idx in range(min_keep, len(normalized)):
            current = normalized[idx]
            previous = normalized[idx - 1]
            drop = previous - current
            if current >= floor and drop <= 0.25 and keep < max_keep:
                keep += 1
                continue
            break

        return max(1, keep)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    _STOPWORDS = {
        "is", "are", "the", "a", "an", "of", "in", "on", "at", "to", "do", "does",
        "there", "any", "show", "shows", "shown", "see", "can", "you", "what",
        "which", "who", "where", "when", "this", "that", "video", "frame", "screen",
        "and", "or", "with", "have", "has", "it", "i", "me", "my", "be", "был",
    }

    def _keyword_caption_search(
        self,
        question: str,
        *,
        video_id: Optional[str],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Fallback retrieval: match query content-words against indexed captions
        and OCR text. CLIP vector similarity is weak on stylized/animated frames,
        but the caption we extracted at index time often names the object
        directly (e.g. caption 'Owls and otters' for the query 'is there an owl').
        """
        if not hasattr(self.frame_index, "list_frames"):
            return []
        terms = [
            t.strip(".,!?'\"").lower()
            for t in question.split()
            if len(t) > 2 and t.strip(".,!?'\"").lower() not in self._STOPWORDS
        ]
        if not terms:
            return []
        # singular/plural-tolerant matching (owl <-> owls)
        def _matches(text: str) -> int:
            text = (text or "").lower()
            hits = 0
            for term in terms:
                if term in text or (term + "s") in text or term.rstrip("s") in text:
                    hits += 1
            return hits

        rows = self.frame_index.list_frames(video_id=video_id, limit=100000)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for r in rows:
            blob = " ".join([
                r.get("caption", "") or "",
                r.get("ocr_text", "") or "",
                r.get("board_text", "") or "",
                r.get("screen_text", "") or "",
                " ".join(r.get("tags", []) or []),
                " ".join(r.get("detected_objects", []) or []),
                r.get("spatial_layout", "") or "",
            ])
            h = _matches(blob)
            if h > 0:
                r = dict(r)
                # caption keyword hits are high-precision for object presence;
                # give them a strong score so they survive _adaptive_answer_count
                r["score"] = 0.55 + 0.1 * min(h, 4)
                r["_via"] = "caption_keyword"
                scored.append((h, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.clip is None:
            raise RuntimeError("CLIP embedder is required for visual retrieval")
        k = int(top_k or self.top_k_retrieve)
        q_vec = self.clip.embed_text(question)
        results = self.frame_index.search_by_vector(q_vec, top_k=k, video_id=video_id)

        # Merge in caption/OCR keyword matches. These rescue object queries that
        # CLIP ranks poorly on stylized frames. Dedupe by chunk_id, keep the
        # higher score.
        kw = self._keyword_caption_search(question, video_id=video_id)
        if kw:
            by_id: Dict[str, Dict[str, Any]] = {}
            for r in results + kw:
                cid = r.get("chunk_id", "")
                if not cid:
                    continue
                prev = by_id.get(cid)
                if prev is None or float(r.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    by_id[cid] = r
            results = sorted(
                by_id.values(), key=lambda r: float(r.get("score", 0.0)), reverse=True
            )

        logger.info(
            "VQA retrieve: q=%r returned %d candidates (%d via caption-kw) (video_id=%s)",
            question[:80],
            len(results),
            len(kw),
            video_id,
        )
        return results

    # ------------------------------------------------------------------
    # Single-question answer
    # ------------------------------------------------------------------

    def _caption_fallback(
        self,
        question: str,
        chosen: List[Dict[str, Any]],
        *,
        video_id: Optional[str],
        started: float,
        include_retrieval: bool,
    ) -> Dict[str, Any]:
        joined = "\n".join(
            f"- [{c['timestamp']:.1f}s] {c.get('caption', '') or c.get('ocr_text', '')}"
            for c in chosen
        )
        return {
            "question": question,
            "video_id": video_id,
            "answer": "Frames not on disk. Based on cached captions:\n" + joined,
            "confidence": 0.4,
            "frames": chosen if include_retrieval else [],
            "elapsed_s": round(time.time() - started, 3),
            "backend": self.vlm.backend_name,
            "model": self.vlm.model_name,
            "note": "caption-only fallback (no frame paths)",
            "mode": "vision_ask",
        }

    def answer(
        self,
        question: str,
        *,
        video_id: Optional[str] = None,
        top_n: Optional[int] = None,
        include_retrieval: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        candidates = self.retrieve(question, video_id=video_id)
        if not candidates:
            has_any_visual = False
            try:
                if hasattr(self.frame_index, "list_frames"):
                    has_any_visual = bool(self.frame_index.list_frames(video_id=video_id, limit=1))
            except Exception:
                has_any_visual = False

            answer = (
                "No relevant visual frames were found for this question in the indexed video. "
                "For speaker, transcript, or topic questions, use the standard /ask pipeline."
                if has_any_visual
                else "No visual content indexed yet — please ingest a video first."
            )
            return {
                "question": question,
                "video_id": video_id,
                "answer": answer,
                "confidence": 0.0,
                "frames": [],
                "elapsed_s": round(time.time() - t0, 3),
                "backend": self.vlm.backend_name,
                "model": self.vlm.model_name,
                "mode": "vision_ask",
            }

        chosen = candidates[: self._adaptive_answer_count(candidates, top_n)]
        paths = [c["frame_path"] for c in chosen if c.get("frame_path")]
        timestamps = [c["timestamp"] for c in chosen if c.get("frame_path")]
        # Carry the indexed caption / OCR text for each chosen frame so the
        # answering VLM has the text we already extracted at index time.
        frame_captions = [
            {
                "timestamp": c.get("timestamp", 0.0),
                "caption": c.get("caption", "") or "",
                "ocr_text": c.get("ocr_text", "") or c.get("all_visible_text", "") or "",
                "objects": c.get("detected_objects", []) or [],
                "spatial_layout": c.get("spatial_layout", "") or "",
                "object_counts": c.get("object_counts", {}) or {},
            }
            for c in chosen if c.get("frame_path")
        ]

        if not paths:
            return self._caption_fallback(
                question,
                chosen,
                video_id=video_id,
                started=t0,
                include_retrieval=include_retrieval,
            )

        vqa_result: VQAResult = self.vlm.vqa(
            paths,
            question,
            video_id=video_id,
            timestamps=timestamps,
            frame_captions=frame_captions,
        )
        return {
            "question": question,
            "video_id": video_id,
            "answer": vqa_result.answer,
            "confidence": vqa_result.confidence,
            "frames": chosen if include_retrieval else [],
            "frames_used": [
                {"path": p, "timestamp": t} for p, t in vqa_result.frames_used
            ],
            "elapsed_s": round(time.time() - t0, 3),
            "backend": vqa_result.backend,
            "model": vqa_result.model_name,
            "mode": "vision_ask",
        }

    # ------------------------------------------------------------------
    # Batch-question answer
    # ------------------------------------------------------------------

    def _extract_json_payload(self, text: str) -> Any:
        import json
        text = (text or "").strip()
        if not text:
            return None
        candidates = [text]
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{") : text.rfind("}") + 1])
        if "[" in text and "]" in text:
            candidates.append(text[text.find("[") : text.rfind("]") + 1])
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _build_batch_prompt(self, questions: Sequence[str]) -> str:
        question_lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        return (
            "Answer ALL user questions about the provided frames in ONE response. "
            "Return STRICT JSON only with this schema: "
            '{"answers":[{"index":1,"question":"...","answer":"...","confidence":0.0}]}\n\n'
            "Rules:\n"
            "- Keep the same order as the questions.\n"
            "- Confidence must be between 0 and 1.\n"
            "- If a question cannot be answered from the frames, say so clearly and use a low confidence.\n\n"
            f"Questions:\n{question_lines}"
        )

    def _parse_batch_answers(
        self,
        raw_text: str,
        questions: Sequence[str],
    ) -> List[Dict[str, Any]]:
        payload = self._extract_json_payload(raw_text)
        rows = []
        if isinstance(payload, dict):
            rows = payload.get("answers", []) or payload.get("items", []) or []
        elif isinstance(payload, list):
            rows = payload

        normalized: List[Dict[str, Any]] = []
        for idx, question in enumerate(questions, start=1):
            found = None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_index = row.get("index")
                row_question = row.get("question")
                if row_index == idx or row_question == question:
                    found = row
                    break
            if found is None:
                normalized.append(
                    {
                        "index": idx,
                        "question": question,
                        "answer": "The model did not return a structured answer for this question.",
                        "confidence": 0.0,
                    }
                )
            else:
                try:
                    conf = float(found.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                normalized.append(
                    {
                        "index": idx,
                        "question": question,
                        "answer": str(found.get("answer", "")).strip() or "No answer.",
                        "confidence": max(0.0, min(conf, 1.0)),
                    }
                )
        return normalized

    def batch_answer(
        self,
        questions: Sequence[str],
        *,
        video_id: Optional[str] = None,
        top_n: Optional[int] = None,
        include_retrieval: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        clean_questions = [str(q).strip() for q in questions if str(q).strip()]
        if not clean_questions:
            return {
                "questions": [],
                "video_id": video_id,
                "answers": [],
                "frames": [],
                "elapsed_s": 0.0,
                "backend": self.vlm.backend_name,
                "model": self.vlm.model_name,
                "mode": "vision_batch_ask",
            }

        retrieval_query = "\n".join(clean_questions)
        candidates = self.retrieve(retrieval_query, video_id=video_id)
        if not candidates:
            return {
                "questions": clean_questions,
                "video_id": video_id,
                "answers": [
                    {
                        "index": i + 1,
                        "question": q,
                        "answer": "No visual content indexed yet — please ingest a video first.",
                        "confidence": 0.0,
                    }
                    for i, q in enumerate(clean_questions)
                ],
                "frames": [],
                "elapsed_s": round(time.time() - t0, 3),
                "backend": self.vlm.backend_name,
                "model": self.vlm.model_name,
                "mode": "vision_batch_ask",
            }

        chosen = candidates[: self._adaptive_answer_count(candidates, top_n)]
        paths = [c["frame_path"] for c in chosen if c.get("frame_path")]
        timestamps = [c["timestamp"] for c in chosen if c.get("frame_path")]

        if not paths:
            answers = []
            for i, q in enumerate(clean_questions, start=1):
                joined = "\n".join(
                    f"- [{c['timestamp']:.1f}s] {c.get('caption', '') or c.get('ocr_text', '')}"
                    for c in chosen
                )
                answers.append(
                    {
                        "index": i,
                        "question": q,
                        "answer": "Frames not on disk. Based on cached captions:\n" + joined,
                        "confidence": 0.4,
                    }
                )
            return {
                "questions": clean_questions,
                "video_id": video_id,
                "answers": answers,
                "frames": chosen if include_retrieval else [],
                "elapsed_s": round(time.time() - t0, 3),
                "backend": self.vlm.backend_name,
                "model": self.vlm.model_name,
                "mode": "vision_batch_ask",
                "note": "caption-only fallback (no frame paths)",
            }

        batch_prompt = self._build_batch_prompt(clean_questions)
        vqa_result: VQAResult = self.vlm.vqa(
            paths,
            batch_prompt,
            video_id=video_id,
            timestamps=timestamps,
        )
        answers = self._parse_batch_answers(vqa_result.answer, clean_questions)
        return {
            "questions": clean_questions,
            "video_id": video_id,
            "answers": answers,
            "frames": chosen if include_retrieval else [],
            "frames_used": [
                {"path": p, "timestamp": t} for p, t in vqa_result.frames_used
            ],
            "elapsed_s": round(time.time() - t0, 3),
            "backend": vqa_result.backend,
            "model": vqa_result.model_name,
            "mode": "vision_batch_ask",
        }


    # ------------------------------------------------------------------
    # Compatibility helpers for api/main.py vision endpoints
    # ------------------------------------------------------------------

    def list_frames(
        self,
        *,
        video_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if not hasattr(self.frame_index, "_collect_visual_metadata"):
            return []
        rows = self.frame_index._collect_visual_metadata(video_id=video_id)
        rows.sort(key=lambda r: float(r.get("start_time") or 0.0))
        out: List[Dict[str, Any]] = []
        for meta in rows[: max(0, int(limit))]:
            chunk_meta = meta.get("metadata", {}) or {}
            out.append({
                "chunk_id":    meta.get("chunk_id", ""),
                "video_id":    meta.get("video_id", ""),
                "frame_path":  chunk_meta.get("frame_path", ""),
                "timestamp":   float(meta.get("start_time") or 0.0),
                "caption":     chunk_meta.get("caption") or chunk_meta.get("short_caption") or "",
                "board_text":  chunk_meta.get("board_text", "") or "",
                "screen_text": chunk_meta.get("screen_text", "") or "",
                "tags":        chunk_meta.get("tags", []) or [],
                "detected_objects": chunk_meta.get("detected_objects", []) or [],
                "spatial_layout": chunk_meta.get("spatial_layout", "") or "",
                "object_counts": chunk_meta.get("object_counts", {}) or {},
                "ocr_text":    chunk_meta.get("ocr_text", "") or "",
                "chunk_type":  "visual",
                "metadata":    chunk_meta,
            })
        return out

    def get_frame(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        store = getattr(self.frame_index, "_store", None) or self.frame_index
        meta_store: Dict[str, Dict[str, Any]] = getattr(store, "_meta", {})
        meta = meta_store.get(chunk_id)
        if not meta or meta.get("chunk_type") != "visual":
            return None
        chunk_meta = meta.get("metadata", {}) or {}
        return {
            "chunk_id":    meta.get("chunk_id", chunk_id),
            "video_id":    meta.get("video_id", ""),
            "frame_path":  chunk_meta.get("frame_path", ""),
            "timestamp":   float(meta.get("start_time") or 0.0),
            "caption":     chunk_meta.get("caption") or chunk_meta.get("short_caption") or "",
            "board_text":  chunk_meta.get("board_text", "") or "",
            "screen_text": chunk_meta.get("screen_text", "") or "",
            "tags":        chunk_meta.get("tags", []) or [],
            "detected_objects": chunk_meta.get("detected_objects", []) or [],
            "spatial_layout": chunk_meta.get("spatial_layout", "") or "",
            "object_counts": chunk_meta.get("object_counts", {}) or {},
            "ocr_text":    chunk_meta.get("ocr_text", "") or "",
            "chunk_type":  "visual",
            "metadata":    chunk_meta,
        }

    def get_stats(self) -> Dict[str, Any]:
        if not hasattr(self.frame_index, "_collect_visual_metadata"):
            return {"total_frames": 0, "clip_faiss_vectors": 0, "videos": {}}
        rows = self.frame_index._collect_visual_metadata(video_id=None)
        per_video: Dict[str, int] = {}
        for row in rows:
            vid = str(row.get("video_id", "") or "")
            per_video[vid] = per_video.get(vid, 0) + 1
        clip_total = 0
        vi = getattr(self.frame_index, "_vision", None)
        if vi and getattr(vi, "_clip_faiss", None):
            clip_total = vi._clip_faiss.ntotal
        return {
            "total_frames": len(rows),
            "clip_faiss_vectors": clip_total,
            "videos": dict(sorted(per_video.items())),
        }

    def delete_video(self, video_id: str) -> int:
        logger.warning(
            "VQAEngine.delete_video(%s): use VideoStore delete path instead.",
            video_id,
        )
        return 0


__all__ = ["VQAEngine", "VisualStoreAdapter", "FrameIndex", "InMemoryFrameIndex"]