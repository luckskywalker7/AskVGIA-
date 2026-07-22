"""
core/vlm_contracts.py
=====================

Dataclasses & type contracts for the Vision-Language Model (VLM) subsystem.

Replaces the legacy PaddleOCR-only flow with a richer representation:
    - Natural-language keyframe descriptions
    - CLIP-style multimodal embeddings
    - VQA (Visual Question Answering) results
    - OCR text as a PRIMARY signal for whiteboard/screen frames
    - scene_change_score per frame for timeline navigation

IMPROVED v2.0:
    - KeyframeDescription.scene_change_score  (float 0→1)
    - KeyframeDescription.is_major_change     (bool)
    - KeyframeDescription.board_text          (str) — dedicated field for
      whiteboard / slide / screen text, separate from caption
    - KeyframeDescription.text_confidence     (float) — OCR / board text confidence
    - TimestampIndex — global timestamp→frame_path lookup exported with video
    - VQAResult.source_frames_timestamps improved

Author: GraphRAG Team
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VLMBackend(str, Enum):
    """Supported VLM execution backends."""
    OLLAMA = "ollama"              # local via Ollama daemon (recommended)
    TRANSFORMERS = "transformers"  # direct HuggingFace Transformers
    OPENAI = "openai"              # OpenAI-compatible API (vLLM, LMStudio...)
    MOCK = "mock"                  # no-op backend for CI/tests


class VLMModel(str, Enum):
    """Canonical names of supported VLMs with known VRAM envelopes."""
    QWEN25_VL_7B = "qwen2.5vl:7b"         # ~12 GB fp16, ~8 GB int4 — best quality
    QWEN25_VL_3B = "qwen2.5vl:3b"         # ~6 GB — good multilingual
    INTERNVL3_8B = "internvl3:8b"         # strong VQA
    SMOLVLM_22B  = "smolvlm:2.2b"         # ~4 GB — fast fallback
    MOONDREAM2   = "moondream:latest"           # ~2 GB — CPU-friendly
    MINICPM_V_26 = "minicpm-v:2.6"        # ~8 GB — strong at text extraction


class FrameKind(str, Enum):
    """How a keyframe was selected from the video timeline."""
    INTERVAL = "interval"                    # fixed interval sampling
    SCENE_CHANGE = "scene_change"            # small scene change
    MAJOR_SCENE_CHANGE = "major_scene_change"  # NEW: big/significant change
    MOTION_PEAK = "motion_peak"              # high-motion moment
    CUSTOM = "custom"                        # caller-provided timestamp


class FrameContentType(str, Enum):
    """What kind of visual content this frame primarily contains."""
    LECTURE = "lecture"           # person talking / presenting
    WHITEBOARD = "whiteboard"     # handwritten content on board
    SLIDE = "slide"               # slide / PowerPoint / presentation
    SCREEN = "screen"             # computer screen / code / UI
    DIAGRAM = "diagram"           # chart / graph / schema
    MIXED = "mixed"               # mix of the above
    GENERAL = "general"           # none of the above


# ---------------------------------------------------------------------------
# Timestamp index — exported from VisionIndexer alongside chunks
# ---------------------------------------------------------------------------


@dataclass
class TimestampEntry:
    """One entry in the timestamp index."""
    timestamp_s: float
    frame_index: int
    frame_path: str
    chunk_id: str         # corresponding VisualChunk.chunk_id
    kind: str             # FrameKind value
    scene_change_score: float = 0.0
    is_major_change: bool = False
    content_type: str = FrameContentType.GENERAL.value
    caption: str = ""


@dataclass
class VideoTimestampIndex:
    """
    Dense mapping: timestamp_s → TimestampEntry.

    This is stored in the vision JSON and in memory so the search layer
    can answer "what was shown at MM:SS?" without re-running the VLM.
    """
    video_id: str
    fps: float
    total_frames: int
    duration_s: float
    entries: List[TimestampEntry] = field(default_factory=list)

    # ---- lookup helpers -----------------------------------------------

    def at(self, timestamp_s: float, tolerance_s: float = 2.0) -> Optional[TimestampEntry]:
        """
        Return the nearest TimestampEntry within tolerance_s of timestamp_s.
        Returns None if no entry is within tolerance.
        """
        if not self.entries:
            return None
        best = min(self.entries, key=lambda e: abs(e.timestamp_s - timestamp_s))
        if abs(best.timestamp_s - timestamp_s) <= tolerance_s:
            return best
        return None

    def in_range(self, start_s: float, end_s: float) -> List[TimestampEntry]:
        """Return all entries in [start_s, end_s]."""
        return [e for e in self.entries if start_s <= e.timestamp_s <= end_s]

    def major_changes(self) -> List[TimestampEntry]:
        """Return only major scene change entries."""
        return [e for e in self.entries if e.is_major_change]

    def by_content_type(self, ct: str) -> List[TimestampEntry]:
        return [e for e in self.entries if e.content_type == ct]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_s": self.duration_s,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoTimestampIndex":
        entries = [TimestampEntry(**e) for e in d.get("entries", [])]
        return cls(
            video_id=d["video_id"],
            fps=d.get("fps", 25.0),
            total_frames=d.get("total_frames", 0),
            duration_s=d.get("duration_s", 0.0),
            entries=entries,
        )


# ---------------------------------------------------------------------------
# Keyframe descriptions
# ---------------------------------------------------------------------------


@dataclass
class KeyframeDescription:
    """
    Rich description of a single video keyframe produced by a VLM.

    IMPROVED v2.0:
        - board_text          : text extracted from whiteboard/slide/screen
                                (dedicated field, separate from caption OCR)
        - text_confidence     : confidence of board_text / detected_text
        - content_type        : FrameContentType enum value
        - scene_change_score  : Bhattacharyya distance from previous frame
        - is_major_change     : True when scene_change_score > threshold
        - all_visible_text    : combined board_text + detected_text deduplicated
    """
    video_id: str
    timestamp: float                           # seconds from video start
    frame_index: int                           # integer frame position
    frame_kind: FrameKind = FrameKind.INTERVAL
    caption: str = ""                          # free-form natural language
    short_caption: str = ""                    # <=15 words summary
    tags: List[str] = field(default_factory=list)
    detected_objects: List[str] = field(default_factory=list)
    object_counts: Dict[str, int] = field(default_factory=dict)  # {"owl": 3, ...}
    spatial_layout: str = ""                    # foreground/background, left/right
    actions: List[str] = field(default_factory=list)  # NEW: what happens/changes (Vgent-style)
    detected_text: str = ""                    # OCR / VLM-extracted raw text
    board_text: str = ""                       # NEW: text on whiteboard/slide/screen
    screen_text: str = ""                      # OCR engine: all on-screen text (PaddleOCR/EasyOCR)
    text_confidence: float = 1.0               # NEW: confidence of board_text
    content_type: str = FrameContentType.GENERAL.value  # NEW: FrameContentType value
    language_hint: Optional[str] = None
    scene_summary: str = ""
    scene_change_score: float = 0.0            # NEW: 0→1, Bhattacharyya distance
    is_major_change: bool = False              # NEW
    embedding: Optional[List[float]] = None
    embedding_dim: Optional[int] = None
    model_name: str = ""
    backend: str = ""
    confidence: float = 1.0
    frame_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ---------- helpers ------------------------------------------------------

    @property
    def doc_id(self) -> str:
        """Stable identifier used for FAISS / graph nodes."""
        key = f"{self.video_id}::{self.frame_index}::{self.timestamp:.3f}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

    @property
    def all_visible_text(self) -> str:
        """
        Combine board_text (VLM), screen_text (OCR engine), and detected_text,
        deduplicated. board_text has priority, then screen_text, then detected_text.
        """
        seen: set = set()
        parts: List[str] = []
        for chunk in [self.board_text, self.screen_text, self.detected_text]:
            if chunk and chunk.strip() not in seen:
                seen.add(chunk.strip())
                parts.append(chunk.strip())
        return "\n".join(parts)

    @property
    def searchable_text(self) -> str:
        """
        Concatenation used for BM25 / full-text indexing.
        Board/screen text is given high priority so retrieval works well
        for "what is written at time XX:XX?" style questions.
        """
        parts: List[str] = []

        # Priority 1: VLM board text (whiteboard / slide / screen — context-aware)
        if self.board_text:
            parts.append(self.board_text)

        # Priority 2: OCR engine screen text (character-precise, Arabic/French safe)
        if self.screen_text and self.screen_text != self.board_text:
            parts.append(self.screen_text)

        # Priority 3: short caption
        if self.short_caption:
            parts.append(self.short_caption)

        # Priority 4: full caption
        if self.caption and self.caption != self.short_caption:
            parts.append(self.caption)

        # Priority 5: tags & objects
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.detected_objects:
            parts.append(" ".join(self.detected_objects))

        # Priority 5b: spatial layout — lets "what is behind/left of X" retrieve
        if self.spatial_layout:
            parts.append(self.spatial_layout)

        # Priority 6: other detected text (legacy fallback)
        if self.detected_text and self.detected_text not in (self.board_text, self.screen_text):
            parts.append(self.detected_text)

        return "\n".join(p.strip() for p in parts if p.strip())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["frame_kind"] = (
            self.frame_kind.value
            if isinstance(self.frame_kind, FrameKind)
            else self.frame_kind
        )
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KeyframeDescription":
        d = dict(d)
        fk = d.get("frame_kind", "interval")
        try:
            d["frame_kind"] = FrameKind(fk)
        except Exception:
            d["frame_kind"] = FrameKind.INTERVAL
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Visual Question Answering
# ---------------------------------------------------------------------------


@dataclass
class VQAResult:
    """Answer from the VLM about one or more frames."""
    question: str
    answer: str
    video_id: Optional[str] = None
    frames_used: List[Tuple[str, float]] = field(default_factory=list)
    # ^^ list of (frame_path, timestamp_s)
    confidence: float = 1.0
    model_name: str = ""
    backend: str = ""
    elapsed_s: float = 0.0
    rationale: str = ""
    # NEW: which timestamps were examined
    timestamps_examined: List[float] = field(default_factory=list)
    # NEW: whether this was a temporal query ("at 01:30")
    is_temporal_query: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VLMConfig:
    """
    Runtime configuration for the VLM subsystem.

    IMPROVED v2.0:
        - major_scene_threshold  : score above which a change is "major"
        - board_text_enabled     : whether to run dedicated whiteboard/slide OCR
        - content_type_detection : whether to classify frame content type
        - temporal_vqa_window_s  : seconds around a timestamp to include
                                   when answering temporal questions
    """
    enabled: bool = True

    # Primary & fallback models
    primary_model: str = VLMModel.QWEN25_VL_7B.value
    fallback_models: List[str] = field(default_factory=lambda: [
        VLMModel.SMOLVLM_22B.value,
        VLMModel.MOONDREAM2.value,
    ])

    # Backend wiring
    backend: str = VLMBackend.OLLAMA.value
    ollama_base_url: str = "http://localhost:11434"
    transformers_device: str = "auto"
    transformers_dtype: str = "auto"
    openai_base_url: str = ""
    openai_api_key: str = ""

    # ---- Prompts -----------------------------------------------------------

    caption_prompt: str = (
        "You are a strict video frame analyst. Your only job is to describe what you "
        "can DIRECTLY SEE in this image. You must NEVER guess, infer, or imagine "
        "content that is not clearly visible.\n\n"
        "STRICT RULES:\n"
        "1. NEVER name a real person unless their name is explicitly written on screen "
        "(name badge, lower-third, subtitle, or text overlay). If you see a face, "
        "describe ONLY their visible appearance: age, gender, hair color, clothing. "
        "Do NOT guess who they are.\n"
        "2. NEVER add context, symbolism, or interpretation. Describe only literal "
        "visible elements.\n"
        "3. If something is unclear or partially visible, say 'partially visible' or "
        "'unclear'. Never fill gaps with assumptions.\n"
        "4. For animated/illustrated content, describe the illustration literally.\n\n"
        "FEW-SHOT EXAMPLES:\n\n"
        "WRONG: 'Barack Obama is giving a speech about climate change.'\n"
        "RIGHT: 'A man in a dark suit stands at a podium. His face is not clearly "
        "visible. A microphone is in front of him.'\n\n"
        "WRONG: 'Einstein is explaining the theory of relativity on a whiteboard.'\n"
        "RIGHT: 'An elderly white-haired man in a grey sweater stands next to a "
        "blackboard. The blackboard has mathematical equations written on it.'\n\n"
        "WRONG: 'A cartoon character symbolizing hope and dreams floats on a cloud.'\n"
        "RIGHT: 'An animated illustration shows a cartoon figure with brown skin lying "
        "on a white fluffy cloud against a blue sky background.'\n\n"
        "WRONG: 'The CEO of Tesla discusses electric vehicles.'\n"
        "RIGHT: 'A man with short brown hair sits in a studio chair. He appears to be "
        "speaking. No name is visible on screen.'\n\n"
        "NOW describe this frame (3-5 sentences, only what you directly see):"
    )
    # NOTE: short_caption_prompt, tags_prompt, ocr_prompt, board_text_prompt,
    # content_type_prompt were removed on 2026-06-18 — the combined describe
    # prompt (_COMBINED_PROMPT in vlm_client) produces all these fields in one
    # call, so these standalone prompts had no callers. Archived in
    # ARCHIVED_dead_vlm_prompts.md. caption_prompt kept (combined-call fallback).
    vqa_prompt_template: str = (
        "You are analyzing one or more video frames. Answer the question based on "
        "what is visible, including stylized, animated, cartoon, or illustrated "
        "depictions of objects, animals, and scenes.\n\n"
        "RULES:\n"
        "1. NEVER name a specific real person unless their name is written on "
        "screen. Describe them instead ('a man in a dark suit').\n"
        "2. Animated or simplified art still COUNTS. If the frame shows a "
        "cartoon owl, the answer to 'is there an owl?' is YES. Do not require "
        "photorealism.\n"
        "3. If INDEX-TIME NOTES are provided below and they name an object that "
        "is consistent with the frame, you may rely on them to confirm the "
        "object is present.\n"
        "4. Only say 'not visible in the frame' when neither the image nor the "
        "notes support the object — do NOT use it as a default safe answer.\n"
        "5. For yes/no questions, answer with a clear Yes or No first, then a "
        "short reason.\n"
        "6. Quote any on-screen text exactly as written; do not invent text.\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "Q: Who is speaking in this frame?\n"
        "WRONG: 'Barack Obama is speaking.'\n"
        "RIGHT: 'A man in a dark suit is speaking. No name is visible on screen.'\n\n"
        "Q: Is there an owl on screen?\n"
        "WRONG: 'This is not clearly visible in the frame.' (when a cartoon owl is shown)\n"
        "RIGHT: 'Yes. An illustrated owl is shown among other forest animals.'\n\n"
        "Q: What is written on the board?\n"
        "WRONG: 'The board explains relativity.'\n"
        "RIGHT: 'The board shows: E = mc² and F = ma'\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    temporal_vqa_prompt_template: str = (
        "You are analyzing a video frame captured at {timestamp}.\n\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "1. Describe ONLY what you directly see — no guesses, no outside knowledge.\n"
        "2. NEVER name a real person unless their name appears as text on screen "
        "(subtitle, lower-third, name badge, overlay).\n"
        "3. For animated content, describe the illustration literally — do NOT "
        "interpret symbolism or assign real identities to cartoon figures.\n"
        "4. If something is unclear, say 'not clearly visible'.\n"
        "5. Quote any on-screen text exactly as written.\n\n"
        "FEW-SHOT EXAMPLES (what good answers look like):\n\n"
        "Frame: animated figure lying on a cloud\n"
        "WRONG answer: 'Barack Obama lying on a cloud symbolizing big dreams.'\n"
        "RIGHT answer: 'An animated cartoon figure with dark hair lies on a white "
        "cloud. The style is 2D illustration. No text is visible in the frame.'\n\n"
        "Frame: man at whiteboard with equations\n"
        "WRONG answer: 'Albert Einstein explains his theory at the blackboard.'\n"
        "RIGHT answer: 'A man with grey hair stands at a blackboard. The board shows "\
        "the equation: E = mc². No name is visible on screen.'\n\n\"\n"
        "Frame: person speaking in studio\n"
        "WRONG answer: 'The CEO of a major company discusses quarterly results.'\n"
        "RIGHT answer: 'A woman in a blue blazer sits in a studio chair speaking. "
        "A lower-third text reads: Dr. Sarah Chen, Neuroscientist.'\n\n"
        "Now describe what you see at {timestamp} and answer the question.\n\n"
        "Question: {question}\n\n"
        "Answer (only what is directly visible):"
    )

    # ---- Keyframe extraction -----------------------------------------------
    frame_interval_s: float = 3.0
    max_frames_per_video: int = 200
    scene_threshold: float = 0.30
    major_scene_threshold: float = 0.55        # NEW: threshold for "big" changes
    # FIX: was 5 — far too tight. 5/256 pHash bits means only near-identical
    # frames are de-duped; frames 5 s apart in the same scene collide and get
    # dropped, leaving just 1 frame per video.
    # 18 ≈ 93% visual similarity on a 256-bit pHash — drops true duplicates
    # (same static title card, freeze-frame) without killing distinct shots.
    dedup_phash_distance: int = 18
    structural_sim_threshold: float = 0.97     # FIX: was 0.92, too tight for film
    min_frame_size_px: int = 64

    # ---- New v2.0 features -------------------------------------------------
    board_text_enabled: bool = True            # run dedicated whiteboard/slide OCR
    content_type_detection: bool = True        # classify each frame's content type
    temporal_vqa_window_s: float = 5.0         # seconds around a timestamp for VQA
    scene_change_threshold: float = 0.30       # alias kept for back-compat

    # ---- CLIP embedder -----------------------------------------------------
    clip_enabled: bool = True
    clip_model: str = "sentence-transformers/clip-ViT-B-32"
    clip_embedding_dim: int = 512
    clip_device: str = "auto"

    # ---- Retry / concurrency -----------------------------------------------
    max_retries: int = 2
    request_timeout_s: float = 60.0
    max_concurrency: int = 2

    # ---- OCR fallback ------------------------------------------------------
    ocr_fallback_enabled: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "fr", "ar"])

    # ---------- factory ------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Any) -> "VLMConfig":
        """Build a VLMConfig from the global `settings` object (duck-typed)."""
        def g(name: str, default: Any) -> Any:
            return getattr(settings, name, default)

        return cls(
            enabled=bool(g("vision_model_enabled", True)),
            primary_model=str(g("vlm_primary_model", VLMModel.QWEN25_VL_7B.value)),
            fallback_models=list(g("vlm_fallback_models",
                                   [VLMModel.SMOLVLM_22B.value, VLMModel.MOONDREAM2.value])),
            backend=str(g("vlm_backend", VLMBackend.OLLAMA.value)),
            ollama_base_url=str(g("vlm_ollama_url", "http://localhost:11434")),
            transformers_device=str(g("vlm_device", "auto")),
            transformers_dtype=str(g("vlm_dtype", "auto")),
            openai_base_url=str(g("vlm_openai_base_url", "")),
            openai_api_key=str(g("vlm_openai_api_key", "")),
            frame_interval_s=float(g("vision_frame_interval_s", 3.0)),
            max_frames_per_video=int(g("vision_max_frames", 200)),
            scene_threshold=float(g("vision_scene_threshold", 0.30)),
            major_scene_threshold=float(g("vision_major_scene_threshold", 0.55)),
            dedup_phash_distance=int(g("vision_dedup_phash_distance", 18)),
            structural_sim_threshold=float(g("vision_structural_sim_threshold", 0.97)),
            board_text_enabled=bool(g("vlm_board_text_enabled", True)),
            content_type_detection=bool(g("vlm_content_type_detection", True)),
            temporal_vqa_window_s=float(g("vlm_temporal_vqa_window_s", 5.0)),
            clip_enabled=bool(g("vlm_clip_enabled", True)),
            clip_model=str(g("vlm_clip_model", "sentence-transformers/clip-ViT-B-32")),
            clip_embedding_dim=int(g("vlm_clip_dim", 512)),
            clip_device=str(g("vlm_clip_device", "auto")),
            max_retries=int(g("vlm_max_retries", 2)),
            request_timeout_s=float(g("vlm_timeout_s", 60.0)),
            max_concurrency=int(g("vlm_max_concurrency", 2)),
            ocr_fallback_enabled=bool(g("ocr_enabled", True)),
            ocr_languages=list(g("ocr_languages", ["en", "fr", "ar"])),
        )


__all__ = [
    "VLMBackend", "VLMModel", "FrameKind", "FrameContentType",
    "KeyframeDescription", "VQAResult", "VLMConfig",
    "TimestampEntry", "VideoTimestampIndex",
]