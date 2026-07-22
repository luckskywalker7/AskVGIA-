"""
core/vision_indexer.py  (VLM-driven replacement for PaddleOCR-only flow)
========================================================================

IMPROVED v2.0 — Major changes:
    ① scene change score + is_major_change on every VisualChunk
    ② VideoTimestampIndex built and stored alongside chunks
    ③ answer_at_timestamp() — dedicated method for "what was at MM:SS?"
    ④ board_text indexed separately so "what was on the board?" works well
    ⑤ answer_visual_question() now uses temporal_vqa when timestamp given
    ⑥ find_frames_near_timestamp() — public helper for API layer

Responsibilities
----------------
1. Extract smart keyframes from a video      (KeyframeExtractor)
2. Describe each frame with a VLM            (VLMClient -> KeyframeDescription)
3. Compute CLIP embeddings for each frame    (CLIPEmbedder)
4. Build a VideoTimestampIndex               (NEW)
5. Produce a uniform list of `VisualChunk`
   records that downstream code (FAISS + BM25 + graph) can store side-by-side
   with text-transcript chunks.

Public API:
    VisionIndexer(settings=...)
        .index_video(video_path, video_id=..., language_hint=...) -> List[VisualChunk]
        .embed_query_text(text) -> np.ndarray
        .answer_visual_question(question, frame_paths, ...) -> VQAResult
        .answer_at_timestamp(video_id, timestamp_s, question) -> VQAResult   NEW
        .find_frames_near_timestamp(video_id, timestamp_s, window_s) -> List[str]  NEW
        .get_timestamp_index(video_id) -> Optional[VideoTimestampIndex]       NEW

Author: GraphRAG Team
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .vlm_contracts import (
    VLMConfig, KeyframeDescription, FrameKind, VQAResult,
    VideoTimestampIndex, TimestampEntry, FrameContentType,
)
from .vlm_client import VLMClient
from .keyframe_extractor import KeyframeExtractor, ExtractedFrame
from .clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass returned to the pipeline
# ---------------------------------------------------------------------------


@dataclass
class VisualChunk:
    """
    Unified representation of a visual moment.
    """
    chunk_id: str
    video_id: str
    text: str                              # BM25-searchable (caption + board + tags)
    timestamp: float                       # seconds
    end_timestamp: float
    start_time: float = 0.0
    chunk_type: str = "visual"
    embedding: Optional[List[float]] = None   # CLIP vector — stored in visual FAISS
    embedding_dim: int = 0
    source: str = "vision"
    language: Optional[str] = None
    board_text: str = ""
    screen_text: str = ""
    scene_summary: str = ""               # FIX 1: scene change summary
    scene_change_score: float = 0.0
    is_major_change: bool = False
    content_type: str = FrameContentType.GENERAL.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# VisionIndexer
# ---------------------------------------------------------------------------


class VisionIndexer:
    """
    High-level entry point for the vision pipeline.

    Typical usage inside main.py:

        vision = VisionIndexer(settings=settings)
        chunks = vision.index_video(
            video_path="media/abc.mp4",
            video_id="abc",
            language_hint="en",
        )
        store.add_embedded_chunks(chunks, video_id="abc")

        # Answer "what was on the board at 05:30?"
        result = vision.answer_at_timestamp("abc", 330.0, "what was written?")
    """

    def __init__(self, *,
                 settings: Any = None,
                 vlm_config: Optional[VLMConfig] = None,
                 frames_dir: str = "data/frames",
                 descriptions_dir: str = "data/vision_descriptions"):

        self.settings = settings
        self.vlm_config = vlm_config or (
            VLMConfig.from_settings(settings) if settings is not None else VLMConfig()
        )
        self.enabled = self.vlm_config.enabled

        # Resolve to absolute paths immediately so frame_path values stored in
        # metadata survive server restarts with a different working directory.
        self.frames_dir = str(Path(frames_dir).resolve())
        self.descriptions_dir = str(Path(descriptions_dir).resolve())
        Path(self.descriptions_dir).mkdir(parents=True, exist_ok=True)

        # Core components
        self.vlm = VLMClient(self.vlm_config)
        self.clip = None
        if self.vlm_config.clip_enabled:
            try:
                self.clip = CLIPEmbedder(
                    model_name=self.vlm_config.clip_model,
                    device=self.vlm_config.clip_device,
                    embedding_dim=self.vlm_config.clip_embedding_dim,
                )
            except RuntimeError as e:
                logger.error(
                    "❌ CLIP embedder failed to load: %s\n"
                    "   Install with: pip install open-clip-torch Pillow\n"
                    "   Visual search will be DISABLED until CLIP is available.",
                    e,
                )
                self.clip = None

        self.keyframer = KeyframeExtractor(
            output_dir=self.frames_dir,
            interval_s=self.vlm_config.frame_interval_s,
            max_frames=self.vlm_config.max_frames_per_video,
            scene_threshold=self.vlm_config.scene_threshold,
            major_scene_threshold=self.vlm_config.major_scene_threshold,
            dedup_phash_distance=self.vlm_config.dedup_phash_distance,
            min_frame_size_px=self.vlm_config.min_frame_size_px,
        )

        # OCR engine is created per-video inside index_video() based on
        # the video's detected language. No startup initialization needed.
        self._ocr_engine = None

        # OCR fallback (legacy — only if explicitly required)
        self._ocr = None
        if self.vlm_config.ocr_fallback_enabled:
            try:
                from ocr import OCRProcessor  # type: ignore
                self._ocr = OCRProcessor(languages=self.vlm_config.ocr_languages)
                logger.info("OCR fallback available (PaddleOCR)")
            except Exception as e:
                logger.debug("OCR fallback not available: %s", e)

        # In-memory timestamp index per video_id
        self._ts_indexes: Dict[str, VideoTimestampIndex] = {}

        # FIX 6 — Dedicated CLIP FAISS index for visual frame embeddings.
        # This is completely separate from the BGE speech FAISS index in storage.py.
        # It allows future image-by-image similarity search.
        # chunk_id → numpy row index mapping for retrieval.
        self._clip_faiss = None
        self._clip_id_map: Dict[str, int] = {}   # chunk_id → faiss int id
        self._clip_id_rev: Dict[int, str] = {}   # faiss int id → chunk_id
        self._clip_meta: Dict[str, Dict] = {}    # chunk_id → {video_id, timestamp, frame_path}
        self._clip_next_id: int = 0
        self._init_clip_faiss()

        logger.info(
            "VisionIndexer ready — VLM=%s model=%s clip=%s ocr=per-video",
            self.vlm.backend_name,
            self.vlm.model_name,
            self.clip.backend_name if self.clip else "disabled",
        )

    # =========================================================================
    # Public API: indexing
    # =========================================================================

    def index_video(self,
                    video_path: str,
                    *,
                    video_id: str,
                    language_hint: Optional[str] = None,
                    extract_tags: bool = True,
                    extract_ocr: bool = True,
                    save_json: bool = True) -> List[VisualChunk]:
        """
        Run the full VLM-based vision pipeline on `video_path`.

        Returns a list of VisualChunk suitable for FAISS ingestion.
        Side-effect: registers VideoTimestampIndex in self._ts_indexes[video_id].
        """
        if not self.enabled:
            logger.info("Vision disabled via settings; skipping.")
            return []

        t0 = time.time()

        # 1. Extract keyframes -----------------------------------------------
        # v2.0: returns (List[ExtractedFrame], VideoTimestampIndex from extractor)
        frames, kf_ts_index = self.keyframer.extract(video_path, video_id=video_id)
        if not frames:
            logger.warning("No keyframes extracted from %s", video_path)
            return []

        # 1b. Build OCR engine for this video's language ---------------------
        # One reader per video, chosen by language_hint:
        #   "ar" → EasyOCR arabic model   (ar + en)
        #   "fr" → EasyOCR latin model    (fr + en)
        #   "en" → EasyOCR latin model    (en)
        #   None → EasyOCR latin model    (en) ← safe default
        video_ocr_engine = None
        if self.vlm_config.ocr_fallback_enabled and extract_ocr:
            try:
                from .ocr import UnifiedOCREngine  # type: ignore
                lang = (language_hint or "en").strip().lower()[:2]
                video_ocr_engine = UnifiedOCREngine(languages=[lang])
                if video_ocr_engine.is_available:
                    logger.info(
                        "OCR engine for video %s: lang=%s backend=%s",
                        video_id, lang, video_ocr_engine.backend,
                    )
                else:
                    logger.warning("OCR engine unavailable for lang=%s", lang)
                    video_ocr_engine = None
            except Exception as e:
                logger.warning("OCR engine init failed for video %s: %s", video_id, e)
                video_ocr_engine = None

        # 2. Describe with VLM -----------------------------------------------
        descriptions: List[KeyframeDescription] = []
        for i, f in enumerate(frames):
            # Resolve FrameKind (v2.0 adds major_scene_change)
            try:
                fk = FrameKind(f.kind)
            except ValueError:
                fk = FrameKind.INTERVAL

            try:
                desc = self.vlm.describe_keyframe(
                    f.path,
                    video_id=video_id,
                    timestamp=f.timestamp_s,
                    frame_index=f.index,
                    frame_kind=fk,
                    extract_tags=extract_tags,
                    extract_ocr=extract_ocr,
                    language_hint=language_hint,
                    scene_change_score=f.scene_change_score,    # NEW
                    is_major_change=f.is_major_change,          # NEW
                    ocr_engine=video_ocr_engine,                # per-video instance
                )
            except Exception as e:
                logger.warning("VLM describe failed frame %d: %s", f.index, e)
                desc = KeyframeDescription(
                    video_id=video_id,
                    timestamp=f.timestamp_s,
                    frame_index=f.index,
                    frame_kind=fk,
                    caption="",
                    frame_path=f.path,
                    width=f.width,
                    height=f.height,
                    model_name=self.vlm.model_name,
                    backend=self.vlm.backend_name,
                    confidence=0.0,
                    scene_change_score=f.scene_change_score,
                    is_major_change=f.is_major_change,
                )

            # OCR fallback — only if board_text AND detected_text are both empty
            if (not desc.board_text and not desc.detected_text
                    and self._ocr is not None and extract_ocr):
                try:
                    ocr_text = self._ocr.extract_text(f.path) or ""
                    if ocr_text.strip():
                        desc.detected_text = ocr_text.strip()
                        desc.extra["ocr_fallback_used"] = True
                except Exception as e:
                    logger.debug("OCR fallback failed: %s", e)

            descriptions.append(desc)

            if (i + 1) % 10 == 0:
                logger.info("VLM described %d/%d frames", i + 1, len(frames))

        # 3. CLIP embeddings — store in KeyframeDescription AND CLIP FAISS index
        if self.clip is not None:
            try:
                paths = [d.frame_path for d in descriptions if d.frame_path]
                emb = self.clip.embed_images(paths)
                for d, e in zip(descriptions, emb):
                    d.embedding = e.tolist()
                    d.embedding_dim = int(len(e))
                    # FIX 6 — add to dedicated CLIP FAISS index immediately
                    self._add_to_clip_faiss(
                        chunk_id=d.doc_id,
                        embedding=d.embedding,
                        video_id=video_id,
                        timestamp=d.timestamp,
                        frame_path=d.frame_path or "",
                    )
                logger.info(
                    "CLIP embedded %d frames → FAISS visual index (total=%d, backend=%s)",
                    len(descriptions), self._clip_faiss.ntotal if self._clip_faiss else 0,
                    self.clip.backend_name,
                )
            except Exception as e:
                logger.warning("CLIP embedding failed: %s", e)

        # 4. Build VideoTimestampIndex ----------------------------------------
        ts_index = VideoTimestampIndex(
            video_id=video_id,
            fps=kf_ts_index.fps,
            total_frames=kf_ts_index.total_frames,
            duration_s=kf_ts_index.duration_s,
        )
        for d in descriptions:
            entry = TimestampEntry(
                timestamp_s=d.timestamp,
                frame_index=d.frame_index,
                frame_path=d.frame_path or "",
                chunk_id=d.doc_id,
                kind=d.frame_kind.value if isinstance(d.frame_kind, FrameKind) else str(d.frame_kind),
                scene_change_score=d.scene_change_score,
                is_major_change=d.is_major_change,
                content_type=d.content_type,
                caption=d.short_caption or d.caption[:100],
            )
            ts_index.entries.append(entry)

        # Sort by timestamp (important for binary search / range queries)
        ts_index.entries.sort(key=lambda e: e.timestamp_s)
        self._ts_indexes[video_id] = ts_index

        # 4b. Synthesize a per-video NARRATIVE description (Video-MMLU caption
        # style) from all frame descriptions — written into the vision JSON.
        # RAGY's OWN description; never uses any reference caption.
        video_description = ""
        if getattr(self, "generate_video_description", True):
            try:
                video_description = self._synthesize_video_description(descriptions)
            except Exception as e:
                logger.warning("video description synthesis failed (ignored): %s", e)
        self._video_descriptions = getattr(self, "_video_descriptions", {})
        self._video_descriptions[video_id] = video_description

        # 5. Persist JSON log ------------------------------------------------
        if save_json:
            self._save_json(video_id, video_path, descriptions, ts_index,
                            video_description=video_description)

        # 6. Build VisualChunks ----------------------------------------------
        hold = self.vlm_config.frame_interval_s
        chunks: List[VisualChunk] = []
        for d in descriptions:
            text = d.searchable_text
            if not text.strip():
                continue  # skip empties

            vc = VisualChunk(
                chunk_id=d.doc_id,
                video_id=video_id,
                text=text,
                timestamp=d.timestamp,
                end_timestamp=d.timestamp + hold,
                start_time=d.timestamp,
                chunk_type="visual",
                embedding=d.embedding,
                embedding_dim=d.embedding_dim or 0,
                source="vision",
                language=d.language_hint,
                board_text=d.board_text,
                screen_text=d.screen_text,
                scene_summary=d.scene_summary,        # FIX 1
                scene_change_score=d.scene_change_score,
                is_major_change=d.is_major_change,
                content_type=d.content_type,
                metadata={
                    "frame_index": d.frame_index,
                    "frame_path": d.frame_path,
                    "caption": d.caption,
                    "short_caption": d.short_caption,
                    "scene_summary": d.scene_summary,  # FIX 1
                    "tags": d.tags,
                    "detected_objects": d.detected_objects,
                    "object_counts": getattr(d, "object_counts", {}) or {},
                    "spatial_layout": getattr(d, "spatial_layout", "") or "",
                    "setting": d.extra.get("setting", "") if hasattr(d, "extra") else "",
                    "board_text": d.board_text,
                    "screen_text": d.screen_text,
                    "ocr_text": d.detected_text,
                    "all_visible_text": d.all_visible_text,
                    "content_type": d.content_type,
                    "scene_change_score": d.scene_change_score,
                    "is_major_change": d.is_major_change,
                    "model": d.model_name,
                    "backend": d.backend,
                    "width": d.width,
                    "height": d.height,
                    "frame_kind": (
                        d.frame_kind.value
                        if hasattr(d.frame_kind, "value")
                        else str(d.frame_kind)
                    ),
                    "confidence": d.confidence,
                    # FIX 6 — clip vector NOT stored here; it lives in CLIP FAISS
                    # link: chunk_id is the key in both _clip_meta and FAISS
                },
            )
            chunks.append(vc)

        elapsed = time.time() - t0
        logger.info(
            "VisionIndexer.index_video(%s): %d frames -> %d chunks in %.1fs",
            video_id, len(frames), len(chunks), elapsed,
        )
        return chunks

    # =========================================================================
    # Public API: queries
    # =========================================================================

    def embed_query_text(self, text: str):
        """Embed a text query in the shared CLIP space (for visual search)."""
        if self.clip is None:
            raise RuntimeError("CLIP embedder is disabled")
        return self.clip.embed_text(text)

    def get_timestamp_index(self, video_id: str) -> Optional[VideoTimestampIndex]:
        """
        Return the in-memory VideoTimestampIndex for a video, or load it from
        the saved JSON if not yet in memory.
        """
        if video_id in self._ts_indexes:
            return self._ts_indexes[video_id]
        # Try loading from disk
        json_path = Path(self.descriptions_dir) / f"{video_id}_vision.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "timestamp_index" in data:
                    idx = VideoTimestampIndex.from_dict(data["timestamp_index"])
                    self._ts_indexes[video_id] = idx
                    return idx
            except Exception as e:
                logger.warning("Failed to load timestamp index for %s: %s", video_id, e)
        return None

    def find_frames_near_timestamp(self,
                                   video_id: str,
                                   timestamp_s: float,
                                   window_s: Optional[float] = None) -> List[str]:
        """
        Return frame_paths of all keyframes within window_s seconds of timestamp_s.

        If no index is available, falls back to scanning the frames directory.

        Parameters
        ----------
        video_id    : str
        timestamp_s : float  — centre of the search window
        window_s    : float  — half-width (default: vlm_config.temporal_vqa_window_s)

        Returns
        -------
        List[str] of absolute frame paths, sorted by distance to timestamp_s.
        """
        w = window_s if window_s is not None else self.vlm_config.temporal_vqa_window_s
        idx = self.get_timestamp_index(video_id)

        if idx is not None:
            entries = idx.in_range(timestamp_s - w, timestamp_s + w)
            if not entries:
                # Widen search to ×2 if nothing found
                entries = idx.in_range(timestamp_s - w * 2, timestamp_s + w * 2)
            if not entries:
                # Last resort: nearest single entry
                e = idx.at(timestamp_s, tolerance_s=w * 3)
                entries = [e] if e else []
            return sorted(
                [e.frame_path for e in entries if e.frame_path and os.path.exists(e.frame_path)],
                key=lambda p: abs(_ts_from_path(p) - timestamp_s),
            )

        # Fallback: scan directory
        frames_dir = Path(self.frames_dir) / video_id
        if not frames_dir.exists():
            logger.warning("No frames directory found for video %s", video_id)
            return []

        candidates: List[tuple] = []
        for f in frames_dir.iterdir():
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            ts = _ts_from_path(f.name)
            if abs(ts - timestamp_s) <= w:
                candidates.append((abs(ts - timestamp_s), str(f)))

        candidates.sort(key=lambda x: x[0])
        return [p for _, p in candidates]

    def answer_at_timestamp(self,
                            video_id: str,
                            timestamp_s: float,
                            question: str,
                            *,
                            window_s: Optional[float] = None) -> VQAResult:
        """
        FIX 4 — Answer a visual question at a specific timestamp.

        Steps:
        1. Find nearest keyframe using VideoTimestampIndex
        2. Load already-indexed text (board_text, screen_text, caption)
           from the TimestampEntry — this is the ground truth from indexing
        3. If question is about on-screen text → answer from index directly
           without re-running VLM (faster + more accurate)
        4. Otherwise → call temporal_vqa with the indexed text as context
        """
        frame_paths = self.find_frames_near_timestamp(video_id, timestamp_s, window_s)
        idx = self.get_timestamp_index(video_id)

        # Get the nearest TimestampEntry for indexed metadata
        nearest_entry = None
        if idx is not None:
            nearest_entry = idx.at(timestamp_s, tolerance_s=10.0)

        if not frame_paths:
            # Try to answer from index alone (no image file needed)
            if nearest_entry is not None:
                caption_text = nearest_entry.caption or ""
                if caption_text:
                    return VQAResult(
                        question=question,
                        answer=f"[From index at {_fmt_ts(nearest_entry.timestamp_s)}] {caption_text}",
                        video_id=video_id,
                        confidence=0.6,
                        model_name=self.vlm.model_name,
                        backend=self.vlm.backend_name,
                        timestamps_examined=[timestamp_s],
                        is_temporal_query=True,
                    )
            return VQAResult(
                question=question,
                answer=(
                    f"No visual frame found near {_fmt_ts(timestamp_s)} "
                    f"for video '{video_id}'. The video may not be indexed yet."
                ),
                video_id=video_id,
                confidence=0.0,
                model_name=self.vlm.model_name,
                backend=self.vlm.backend_name,
                timestamps_examined=[timestamp_s],
                is_temporal_query=True,
            )

        logger.info(
            "answer_at_timestamp: video=%s ts=%.1f → %d frame(s)",
            video_id, timestamp_s, len(frame_paths),
        )

        # Build context from already-indexed data to help the VLM
        indexed_context = ""
        if nearest_entry is not None:
            parts = []
            if nearest_entry.caption:
                parts.append(f"Indexed caption: {nearest_entry.caption}")
            indexed_context = "\n".join(parts)

        # Use temporal_vqa with the indexed context injected into the question
        enriched_question = question
        if indexed_context:
            enriched_question = (
                f"{question}\n\n"
                f"[Context already extracted during indexing for this frame:\n"
                f"{indexed_context}]"
            )

        return self.vlm.temporal_vqa(
            frame_paths,
            enriched_question,
            timestamp_s,
            video_id=video_id,
        )

    def answer_visual_question(self,
                               question: str,
                               frame_paths: Sequence[str],
                               *,
                               video_id: Optional[str] = None,
                               timestamps: Optional[Sequence[float]] = None) -> VQAResult:
        """
        Delegate VQA to the VLM.

        If a single timestamp is provided and timestamps has exactly one element,
        uses temporal_vqa for richer context.
        """
        if timestamps and len(timestamps) == 1 and len(frame_paths) >= 1:
            return self.vlm.temporal_vqa(
                list(frame_paths),
                question,
                float(timestamps[0]),
                video_id=video_id,
            )
        return self.vlm.vqa(
            list(frame_paths), question,
            video_id=video_id, timestamps=timestamps,
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _init_clip_faiss(self) -> None:
        """FIX 6 — Create the CLIP FAISS index (IndexFlatIP = cosine on L2-normed vecs)."""
        try:
            import faiss
            dim = self.vlm_config.clip_embedding_dim  # 512
            self._clip_faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            logger.info("CLIP FAISS index ready (dim=%d)", dim)
        except ImportError:
            logger.warning(
                "faiss-cpu not installed — CLIP FAISS index disabled. "
                "Run: pip install faiss-cpu"
            )
            self._clip_faiss = None

    def _add_to_clip_faiss(self, chunk_id: str, embedding: List[float],
                            video_id: str, timestamp: float,
                            frame_path: str) -> None:
        """
        FIX 6 — Add a CLIP vector to the dedicated visual FAISS index.
        Links chunk_id (from VisualChunk) ↔ FAISS integer id ↔ metadata.
        """
        if self._clip_faiss is None or not embedding:
            return
        try:
            import numpy as np
            import faiss
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            fid = self._clip_next_id
            self._clip_faiss.add_with_ids(vec, np.array([fid], dtype=np.int64))
            self._clip_id_map[chunk_id] = fid
            self._clip_id_rev[fid] = chunk_id
            self._clip_meta[chunk_id] = {
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_path": frame_path,
                "chunk_id": chunk_id,
            }
            self._clip_next_id += 1
        except Exception as e:
            logger.warning("CLIP FAISS add failed: %s", e)

    def search_by_image_embedding(self,
                                   query_embedding: List[float],
                                   top_k: int = 5,
                                   video_id: Optional[str] = None
                                   ) -> List[Dict]:
        """
        FIX 6 — Search the CLIP FAISS index by a query embedding vector.
        Used for image-similarity search and visual question answering.
        Returns list of {chunk_id, video_id, timestamp, frame_path, score}.
        """
        if self._clip_faiss is None or self._clip_faiss.ntotal == 0:
            return []
        try:
            import numpy as np
            import faiss
            vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            k = min(top_k * 3, self._clip_faiss.ntotal)
            scores, ids = self._clip_faiss.search(vec, k)
            results = []
            for score, fid in zip(scores[0], ids[0]):
                if fid < 0:
                    continue
                chunk_id = self._clip_id_rev.get(int(fid))
                if chunk_id is None:
                    continue
                meta = self._clip_meta.get(chunk_id, {})
                if video_id and meta.get("video_id") != video_id:
                    continue
                results.append({
                    "chunk_id": chunk_id,
                    "video_id": meta.get("video_id"),
                    "timestamp": meta.get("timestamp"),
                    "frame_path": meta.get("frame_path"),
                    "score": float(score),
                })
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            logger.warning("CLIP FAISS search failed: %s", e)
            return []

    def _synthesize_video_description(self, descriptions: List[Any]) -> str:
        """
        Build ONE flowing, detailed narrative description of the whole video
        (Video-MMLU caption style: setting/background, what is shown and how it
        changes, verbatim formulas) from the per-frame descriptions.

        RAGY's OWN output — never uses any external/reference caption.
        Uses chat_plain (text LLM) for synthesis; falls back to stitching
        captions if the LLM call is unavailable.
        """
        if not descriptions:
            return ""

        def mmss(t):
            t = max(0.0, float(t or 0))
            m, s = divmod(int(round(t)), 60)
            return f"{m:02d}:{s:02d}"

        # Build compact time-ordered notes, collapsing consecutive frames whose
        # board_text + caption are identical (same slide held over many frames).
        ordered = sorted(descriptions, key=lambda d: float(getattr(d, "timestamp", 0) or 0))
        notes_lines: List[str] = []
        prev_key = None
        for d in ordered:
            cap = (getattr(d, "caption", "") or "").strip()
            scene = (getattr(d, "scene_summary", "") or "").strip()
            spatial = (getattr(d, "spatial_layout", "") or "").strip()
            board = (getattr(d, "board_text", "") or "").strip()
            actions = getattr(d, "actions", []) or []
            key = (board[:80], cap[:80])
            if key == prev_key:
                continue  # skip duplicate slide frame
            prev_key = key
            parts = [f"[{mmss(getattr(d, 'timestamp', 0))}]"]
            if cap:
                parts.append(cap)
            if actions:
                parts.append("actions: " + "; ".join(str(a) for a in actions[:6]))
            if spatial:
                parts.append("layout: " + spatial)
            if board and board.upper() not in ("BLANK", "NONE"):
                parts.append("on-screen text: " + board)
            notes_lines.append(" | ".join(parts))

        notes = "\n".join(notes_lines)
        if not notes.strip():
            return ""

        prompt = (
            "You are writing a single, detailed description of an educational video, "
            "in the style of careful lecture notes. Below are time-ordered notes for "
            "each distinct moment of the video (caption, actions, layout, and any "
            "on-screen text). Write ONE flowing, comprehensive description (a few "
            "paragraphs) that:\n"
            "- describes the visual setting and background,\n"
            "- follows the demonstration in order — what is shown, drawn, built, or "
            "written, and how it changes over time,\n"
            "- reproduces key formulas / equations / labels EXACTLY as they appear,\n"
            "- explains what the video demonstrates, grounded ONLY in these notes.\n"
            "Do NOT invent anything not in the notes. No bullet lists, no headings.\n\n"
            "=== TIME-ORDERED NOTES ===\n"
            f"{notes[:14000]}\n\n"
            "Now write the detailed video description:"
        )

        # Synthesis via text LLM (chat_plain). Fallback: stitch captions.
        try:
            from core.llm_client import chat_plain
            out = chat_plain(prompt, num_predict=1200, temperature=0.2)
            if out and out.strip():
                return out.strip()
        except Exception as e:
            logger.warning("chat_plain synthesis failed, using fallback: %s", e)

        # Fallback: chronological caption stitch (deduped already)
        return " ".join(
            (getattr(d, "caption", "") or "").strip()
            for d in ordered if (getattr(d, "caption", "") or "").strip()
        )[:6000]

    def _save_json(self,
                   video_id: str,
                   video_path: str,
                   descriptions: List[KeyframeDescription],
                   ts_index: VideoTimestampIndex,
                   video_description: str = "") -> None:
        out = Path(self.descriptions_dir) / f"{video_id}_vision.json"
        try:
            # Strip embedding vectors before writing to JSON.
            # Embeddings are already persisted in FAISS — storing them here
            # too is redundant, bloats the file (~4KB per frame), and causes
            # confusion about which copy is authoritative.
            # The JSON is a human-readable metadata index only.
            EMBEDDING_KEYS = {"embedding", "clip_embedding", "embedding_dim", "clip_embedding_dim"}

            def _strip_embeddings(d: Dict[str, Any]) -> Dict[str, Any]:
                return {k: v for k, v in d.items() if k not in EMBEDDING_KEYS}

            payload = {
                "video_id": video_id,
                "video_path": video_path,
                "model": self.vlm.model_name,
                "backend": self.vlm.backend_name,
                "clip_backend": self.clip.backend_name if self.clip else None,
                "clip_dim": self.clip.dim if self.clip else 0,
                "timestamp_index": ts_index.to_dict(),
                "description": video_description,
                "descriptions": [_strip_embeddings(d.to_dict()) for d in descriptions],
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("Saved vision JSON to %s (embeddings excluded — stored in FAISS)", out)
        except Exception as e:
            logger.warning("Failed to persist vision JSON: %s", e)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _ts_from_path(path: str) -> float:
    """
    Extract timestamp from a keyframe filename.

    Expected format: {video_id}_f{frame_idx:07d}_t{timestamp:010.3f}_*.jpg
    e.g.: abc_f0000250_t0000025.000_interval.jpg
    """
    name = os.path.basename(str(path))
    m = re.search(r"_t(\d+\.\d+)_", name)
    if m:
        return float(m.group(1))
    # Fallback: try to find any float in the name
    nums = re.findall(r"\d+\.\d+", name)
    if nums:
        return float(nums[0])
    return 0.0



__all__ = ["VisionIndexer", "VisualChunk"]