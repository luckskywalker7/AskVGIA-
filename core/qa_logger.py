"""
core/qa_logger.py
=================
Appends every answered question to a JSON log, grouped by video_id.

Structure written (qa_log.json by default):
{
  "<video_id>": {
    "video_id": "<video_id>",
    "qa": [
      {
        "question": "...",
        "answer": "...",
        "grounded_in": "frames" | "segments" | "both" | "none",
        "n_frames": 3,
        "n_segments": 5,
        "timestamp": "2026-06-18T14:03:22Z"
      },
      ...
    ]
  },
  ...
}

- Appends live: each answered question is added under its video_id immediately.
- Multiple questions for the same video accumulate under the same id.
- Concurrency-safe via a process lock + atomic replace.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _chunk_type_of(item: Any) -> str:
    ct = None
    if isinstance(item, dict):
        ct = item.get("chunk_type")
    else:
        ct = getattr(item, "chunk_type", None)
    return str(ct or "speech")


def classify_grounding(search_results: List[Any]) -> tuple:
    """Return (grounded_in, n_frames, n_segments) from the retrieved results."""
    n_frames = 0
    n_segments = 0
    for r in (search_results or []):
        if _chunk_type_of(r) == "visual":
            n_frames += 1
        else:
            n_segments += 1
    if n_frames and n_segments:
        grounded = "both"
    elif n_frames:
        grounded = "frames"
    elif n_segments:
        grounded = "segments"
    else:
        grounded = "none"
    return grounded, n_frames, n_segments


def log_qa(
    log_path: str,
    video_id: str,
    question: str,
    answer: str,
    search_results: List[Any],
    *,
    extra: Optional[dict] = None,
) -> None:
    """
    Append one Q&A entry under its video_id in the JSON log. Never raises
    (logging must not break the answer stream).
    """
    try:
        if not video_id:
            video_id = "_no_video"
        grounded, n_frames, n_segments = classify_grounding(search_results)
        entry = {
            "question": (question or "").strip(),
            "answer": (answer or "").strip(),
            "grounded_in": grounded,
            "n_frames": n_frames,
            "n_segments": n_segments,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if extra:
            entry.update(extra)

        with _LOCK:
            data = {}
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}  # corrupt/empty file → start fresh (don't crash)

            bucket = data.get(video_id)
            if not isinstance(bucket, dict):
                bucket = {"video_id": video_id, "qa": []}
            bucket.setdefault("qa", []).append(entry)
            data[video_id] = bucket

            # atomic write: tmp file in same dir, then replace
            d = os.path.dirname(os.path.abspath(log_path)) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".qa_log_", dir=d)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, log_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        logger.info("QA logged: video=%s grounded=%s (frames=%d segments=%d)",
                    video_id, grounded, n_frames, n_segments)
    except Exception as e:
        logger.warning("log_qa failed (ignored): %s", e)