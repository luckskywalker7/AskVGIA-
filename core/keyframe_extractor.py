"""
core/keyframe_extractor.py
==========================

Smart keyframe extraction from video files.

Strategies (applied in sequence then merged & deduplicated):
    1. Fixed-interval sampling  (every N seconds)
    2. Scene-change detection   (histogram / frame-diff)
    3. Motion peaks             (optional, via frame diff)

IMPROVED v2.0:
    - scene_change_score stored on each frame (0.0 → 1.0)
    - "major" vs "minor" scene changes classified by threshold
    - Every extracted frame stores a precise timestamp_s (float)
    - Configurable major_scene_threshold for "big" changes
    - Frame index → timestamp mapping exported for downstream use

Deduplication:
    - perceptual hash (aHash) distance
    - optional SSIM distance (if scikit-image available)

Returns a list of ExtractedFrame with precise (frame_index, timestamp_s,
output_path, kind, scene_change_score).

This module has only OpenCV as a hard dependency; everything else is optional.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _CV2_OK = True
except ImportError:
    cv2 = None  # type: ignore
    np = None   # type: ignore
    _CV2_OK = False


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFrame:
    index: int
    timestamp_s: float
    path: str
    kind: str               # "interval" | "scene_change" | "major_scene_change" | "motion_peak"
    width: int = 0
    height: int = 0
    scene_change_score: float = 0.0   # NEW: Bhattacharyya distance 0→1 (higher = more different)
    is_major_change: bool = False      # NEW: True when scene_change_score > major_scene_threshold
    prev_timestamp_s: Optional[float] = None  # NEW: timestamp of previous frame for delta calc


# ---------------------------------------------------------------------------
# Timestamp index — maps frame_index → timestamp_s for all video frames
# ---------------------------------------------------------------------------


@dataclass
class VideoTimestampIndex:
    """
    Lightweight index: frame_index → timestamp_s.
    Produced alongside keyframe extraction so the rest of the pipeline
    can always answer "what is the exact timestamp of frame N?"
    """
    video_id: str
    fps: float
    total_frames: int
    duration_s: float
    keyframe_timestamps: Dict[float, str] = field(default_factory=dict)
    # timestamp_s → kind ("interval" | "scene_change" | "major_scene_change")

    def frame_to_ts(self, frame_index: int) -> float:
        """Convert a frame index to a timestamp in seconds."""
        return frame_index / max(self.fps, 1.0)

    def ts_to_frame(self, timestamp_s: float) -> int:
        """Convert a timestamp in seconds to the nearest frame index."""
        return int(round(timestamp_s * self.fps))

    def nearest_keyframe_ts(self, timestamp_s: float) -> Optional[float]:
        """Return the nearest keyframe timestamp to the given time."""
        if not self.keyframe_timestamps:
            return None
        return min(self.keyframe_timestamps.keys(), key=lambda t: abs(t - timestamp_s))

    def keyframes_in_range(self, start_s: float, end_s: float) -> List[Tuple[float, str]]:
        """Return all keyframe (timestamp, kind) pairs in [start_s, end_s]."""
        return [
            (ts, kind)
            for ts, kind in sorted(self.keyframe_timestamps.items())
            if start_s <= ts <= end_s
        ]


# ---------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------


def _ahash(img_bgr, hash_size: int = 8) -> int:
    """Average hash. Returns integer bit vector."""
    if not _CV2_OK:
        return 0
    small = cv2.resize(img_bgr, (hash_size, hash_size),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    bits = (gray > mean).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _phash(img_bgr, hash_size: int = 16) -> int:
    """
    Perceptual hash (pHash) — much more robust than aHash for detecting
    near-duplicate frames (same slide with tiny cursor movement, same
    speaker with slight head turn, etc.).

    Uses DCT (discrete cosine transform) on a 32x32 grayscale image,
    then takes the top-left hash_size x hash_size low-frequency coefficients.
    Produces a 256-bit hash (hash_size=16) instead of aHash's 64-bit.
    """
    if not _CV2_OK:
        return 0
    # Resize to 32x32 for DCT
    small = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    dct = cv2.dct(gray)
    # Take top-left low-frequency block
    low_freq = dct[:hash_size, :hash_size]
    # Median threshold (more robust than mean for perceptual similarity)
    med = np.median(low_freq)
    bits = (low_freq > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _structural_similarity(frame_a, frame_b, target_size: int = 64) -> float:
    """
    Fast structural similarity between two frames.
    Resizes both to target_size x target_size grayscale, then computes
    normalised cross-correlation. Returns 0.0 (totally different) to
    1.0 (identical). Much faster than full SSIM — ~0.1ms per comparison.
    """
    if not _CV2_OK or frame_a is None or frame_b is None:
        return 0.0
    a = cv2.resize(frame_a, (target_size, target_size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(frame_b, (target_size, target_size), interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ga -= ga.mean()
    gb -= gb.mean()
    norm = (np.linalg.norm(ga) * np.linalg.norm(gb))
    if norm < 1e-8:
        return 1.0  # both essentially blank
    return float(np.sum(ga * gb) / norm)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Histogram comparison
# ---------------------------------------------------------------------------


def _compute_hist(frame_bgr) -> Optional[object]:
    """Compute 8x8x8 normalised BGR histogram."""
    if not _CV2_OK or frame_bgr is None:
        return None
    hist = cv2.calcHist(
        [frame_bgr], [0, 1, 2], None,
        [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist


def _hist_diff(h1, h2) -> float:
    """Bhattacharyya distance in [0, 1]. Higher = more different."""
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class KeyframeExtractor:
    """
    Extract keyframes from a video to individual JPG files.

    Usage:
        kex = KeyframeExtractor(output_dir="data/frames",
                                interval_s=3.0, max_frames=200)
        frames, ts_index = kex.extract("video.mp4", video_id="abc123")

    New in v2.0:
        - Returns (List[ExtractedFrame], VideoTimestampIndex)
        - ExtractedFrame has scene_change_score and is_major_change
        - major_scene_threshold controls what counts as a "big" change
          (default 0.55 — well above typical camera-cut threshold of 0.30)
        - Every extracted frame stores a float timestamp_s (not just int index)
    """

    def __init__(self,
                 output_dir: str = "data/frames",
                 interval_s: float = 3.0,
                 max_frames: int = 200,
                 scene_threshold: float = 0.30,
                 major_scene_threshold: float = 0.55,
                 dedup_phash_distance: int = 12,    # pHash 256-bit: 12 is ~95% visual similarity
                 structural_sim_threshold: float = 0.92,  # skip if > 92% structurally similar
                 min_frame_size_px: int = 64,
                 detect_scene_changes: bool = True,
                 jpeg_quality: int = 90,
                 adaptive_frame_profile: bool = True):
        self.output_dir = output_dir
        self.interval_s = float(max(0.25, interval_s))
        self.max_frames = int(max(1, max_frames))
        self.scene_threshold = float(scene_threshold)
        self.major_scene_threshold = float(major_scene_threshold)
        self.dedup_phash_distance = int(dedup_phash_distance)
        self.structural_sim_threshold = float(structural_sim_threshold)
        self.min_frame_size_px = int(min_frame_size_px)
        self.detect_scene_changes = bool(detect_scene_changes)
        self.jpeg_quality = int(jpeg_quality)
        # Adaptive frame profile: pick interval / scene-threshold / dedup by video
        # DURATION at extract time. Short math-animation videos need dense sampling +
        # a SENSITIVE scene threshold (catch every drawing stage); long lectures need
        # a wide interval + COARSE threshold + dedup (static slides held for seconds).
        self.adaptive_frame_profile = bool(adaptive_frame_profile)
        self.content_dedup = False  # set by _apply_duration_profile; read post-VLM
        # remember configured defaults so the profile starts clean each video
        self._base_interval_s = self.interval_s
        self._base_scene_threshold = self.scene_threshold
        self._base_major_scene_threshold = self.major_scene_threshold

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- adaptive profile -------------------------------------------

    def _apply_duration_profile(self, duration_s: float) -> dict:
        """
        Pick interval / scene-threshold / dedup based on video DURATION.
        Returns a dict describing the chosen profile (also sets self.* knobs).

        Targets ~<=270 frames so ingest stays ~<=2h at ~27s/frame:
          - short  (<5 min):  dense 2s, SENSITIVE scene 0.20, content-dedup OFF
                              (math animation: catch every drawing stage)
          - medium (5-20 min): 5s, normal scene 0.30, content-dedup ON
          - long   (>20 min):  10s, COARSE scene 0.40, content-dedup ON (aggressive)
                              (static slides held for seconds; ignore minor flicker)
        """
        # start from configured defaults each video (no compounding)
        self.interval_s = self._base_interval_s
        self.scene_threshold = self._base_scene_threshold
        self.major_scene_threshold = self._base_major_scene_threshold

        if not self.adaptive_frame_profile or duration_s <= 0:
            return {"profile": "default(off)", "interval_s": self.interval_s,
                    "scene_threshold": self.scene_threshold, "content_dedup": False}

        mins = duration_s / 60.0
        if mins < 5:
            name, iv, scene, major, dedup = "short", 2.0, 0.20, 0.45, False
        elif mins <= 20:
            name, iv, scene, major, dedup = "medium", 5.0, 0.30, 0.55, True
        else:
            name, iv, scene, major, dedup = "long", 10.0, 0.40, 0.60, True

        self.interval_s = float(max(0.25, iv))
        self.scene_threshold = float(scene)
        self.major_scene_threshold = float(major)
        self.content_dedup = dedup  # consumed post-VLM in vision_indexer

        prof = {"profile": name, "duration_min": round(mins, 1),
                "interval_s": self.interval_s, "scene_threshold": self.scene_threshold,
                "major_scene_threshold": self.major_scene_threshold,
                "content_dedup": dedup}
        logger.info("Adaptive frame profile = %s | %s", name, prof)
        return prof

    # ---------- public API --------------------------------------------------

    def extract(
        self,
        video_path: str,
        *,
        video_id: str,
    ) -> Tuple[List[ExtractedFrame], VideoTimestampIndex]:
        """
        Extract keyframes from video_path.

        Returns
        -------
        (frames, ts_index)
            frames    : List[ExtractedFrame] — one entry per saved keyframe
            ts_index  : VideoTimestampIndex  — lookup helpers for the whole video
        """
        if not _CV2_OK:
            raise RuntimeError(
                "OpenCV not available. Install with: `pip install opencv-python-headless`"
            )
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        # ADAPTIVE: choose interval / scene-threshold / dedup by duration BEFORE
        # computing step_frames, so short animations get dense sampling and long
        # lectures get a wide interval (avoids the frame-cap truncation bug).
        self._apply_duration_profile(duration)

        logger.info(
            "Extracting keyframes from %s (fps=%.2f, total=%d, dur=%.1fs)",
            video_path, fps, total_frames, duration,
        )

        step_frames = max(1, int(round(self.interval_s * fps)))

        out_dir = Path(self.output_dir) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted: List[ExtractedFrame] = []
        seen_hashes: List[int] = []
        ts_index = VideoTimestampIndex(
            video_id=video_id,
            fps=fps,
            total_frames=total_frames,
            duration_s=duration,
        )

        prev_hist = None
        prev_ts: Optional[float] = None
        frame_idx = -1
        next_interval_frame = 0
        kept = 0
        _dedup_skipped = 0
        _prev_kept_frame = None  # for structural similarity comparison

        # ── Robustness: detect sequential-decode failure ──────────────────
        # Some MP4s (H.265/HEVC, certain profiles) let OpenCV read the header
        # (fps/total_frames) but fail on cap.read() after the first frame,
        # yielding only 1 keyframe. We detect that and switch to SEEK-BASED
        # sampling, which jumps to target timestamps via CAP_PROP_POS_MSEC and
        # uses a different decode path that usually survives these codecs.
        _seq_failed = False
        _consecutive_read_fail = 0

        while kept < self.max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                _consecutive_read_fail += 1
                # If we fail almost immediately (got <=1 frame), the codec is
                # likely undecodable sequentially → switch to seek mode.
                if frame_idx < 2:
                    _seq_failed = True
                break
            frame_idx += 1

            if (frame.shape[0] < self.min_frame_size_px
                    or frame.shape[1] < self.min_frame_size_px):
                continue

            ts = frame_idx / fps if fps > 0 else 0.0

            is_interval = (frame_idx >= next_interval_frame)
            is_scene_change = False
            is_major_change = False
            scene_score = 0.0

            # ---- Scene-change detection ------------------------------------
            curr_hist = _compute_hist(frame)
            if self.detect_scene_changes and prev_hist is not None:
                scene_score = _hist_diff(prev_hist, curr_hist)
                if scene_score > self.scene_threshold:
                    is_scene_change = True
                if scene_score > self.major_scene_threshold:
                    is_major_change = True

            prev_hist = curr_hist

            if not (is_interval or is_scene_change):
                continue

            # ---- Two-tier perceptual dedup ---------------------------------
            # IMPORTANT: dedup applies ONLY to extra scene-change frames, NOT to
            # the regular time-interval grid. The interval grid guarantees even
            # coverage across the whole video; deduping it can collapse a long
            # animated/dark video to almost nothing (pHash treats many visually
            # distinct dark frames as near-identical). So an interval frame is
            # ALWAYS kept; only bonus scene-change frames are dedup-checked.
            if not is_interval:
                # Tier 1: pHash (256-bit) — catches near-identical frames fast
                h = _phash(frame)
                is_phash_dup = any(
                    _hamming(h, seen_h) <= self.dedup_phash_distance
                    for seen_h in seen_hashes
                )
                if is_phash_dup:
                    _dedup_skipped += 1
                    continue

                # Tier 2: structural similarity against the last KEPT frame
                if _prev_kept_frame is not None:
                    sim = _structural_similarity(frame, _prev_kept_frame)
                    if sim > self.structural_sim_threshold:
                        _dedup_skipped += 1
                        continue
            else:
                # still record the hash for future scene-change dedup
                h = _phash(frame)

            seen_hashes.append(h)
            _prev_kept_frame = frame.copy()  # store for next structural comparison

            # ---- Classify kind --------------------------------------------
            if is_major_change:
                kind = "major_scene_change"
            elif is_scene_change and not is_interval:
                kind = "scene_change"
            else:
                kind = "interval"

            # ---- Write to disk --------------------------------------------
            fname = (
                f"{video_id}"
                f"_f{frame_idx:07d}"
                f"_t{ts:010.3f}"
                f"_sc{scene_score:.3f}"
                f"_{kind}.jpg"
            )
            fpath = out_dir / fname
            cv2.imwrite(
                str(fpath), frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )

            ef = ExtractedFrame(
                index=frame_idx,
                timestamp_s=ts,
                path=str(fpath.resolve()),   # ← always absolute: survives CWD changes
                kind=kind,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                scene_change_score=round(scene_score, 4),
                is_major_change=is_major_change,
                prev_timestamp_s=prev_ts,
            )
            extracted.append(ef)
            ts_index.keyframe_timestamps[ts] = kind
            kept += 1
            prev_ts = ts

            if is_interval:
                next_interval_frame = frame_idx + step_frames

        # ── SEEK-BASED FALLBACK ───────────────────────────────────────────
        # Triggered when sequential read() died after <=1 frame (bad codec).
        # We jump to evenly-spaced timestamps and grab one frame at each,
        # which uses OpenCV's seek decode path and survives many H.265 files.
        if _seq_failed and kept <= 1 and duration > 0:
            logger.warning(
                "Sequential decode failed (got %d frames) — switching to "
                "SEEK-BASED sampling for %s", kept, video_id,
            )
            # clear the single bad frame we may have kept
            extracted.clear()
            ts_index.keyframe_timestamps.clear()
            seen_hashes.clear()
            kept = 0
            _prev_kept_frame = None

            # target ~max_frames evenly spaced across the whole video
            n_targets = self.max_frames
            interval = duration / float(n_targets)
            cap.release()
            cap = cv2.VideoCapture(video_path)

            for i in range(n_targets):
                target_s = i * interval
                cap.set(cv2.CAP_PROP_POS_MSEC, target_s * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if (frame.shape[0] < self.min_frame_size_px
                        or frame.shape[1] < self.min_frame_size_px):
                    continue

                # dedup against kept frames (pHash)
                h = _phash(frame)
                if any(_hamming(h, sh) <= self.dedup_phash_distance
                       for sh in seen_hashes):
                    continue
                seen_hashes.append(h)

                ts = target_s
                fname = (
                    f"{video_id}_f{int(ts*fps):07d}"
                    f"_t{ts:010.3f}_sc0.000_interval.jpg"
                )
                fpath = out_dir / fname
                cv2.imwrite(str(fpath), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                extracted.append(ExtractedFrame(
                    index=int(ts * fps),
                    timestamp_s=ts,
                    path=str(fpath.resolve()),
                    kind="interval",
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    scene_change_score=0.0,
                    is_major_change=False,
                    prev_timestamp_s=prev_ts,
                ))
                ts_index.keyframe_timestamps[ts] = "interval"
                kept += 1
                prev_ts = ts

            logger.info("Seek-based sampling extracted %d frames", kept)

        cap.release()

        n_major = sum(1 for f in extracted if f.is_major_change)
        n_scene = sum(1 for f in extracted if f.kind == "scene_change")
        n_interval = sum(1 for f in extracted if f.kind == "interval")
        logger.info(
            "Keyframe stats: read=%d frames, step_frames=%d, kept=%d "
            "(deduped=%d) for %s",
            frame_idx + 1, step_frames, kept, _dedup_skipped, video_id,
        )

        logger.info(
            "Extracted %d keyframes: %d major_scene_change, %d scene_change, "
            "%d interval → %s",
            len(extracted), n_major, n_scene, n_interval, out_dir,
        )
        return extracted, ts_index


__all__ = ["KeyframeExtractor", "ExtractedFrame", "VideoTimestampIndex"]