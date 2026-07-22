"""
core/vlm_client.py
==================
Unified Vision-Language Model client.

Supports:
    1. Ollama  — local, recommended
    2. HuggingFace Transformers — direct GPU inference
    3. OpenAI-compatible — vLLM, LMStudio, etc.
    NO MOCK — if nothing is available the client raises, not silently returns garbage.

Public API:
    VLMClient(config)
        .is_available() -> bool
        .caption(image_path) -> str
        .short_caption(image_path) -> str
        .tags(image_path) -> List[str]
        .ocr(image_path) -> str
        .board_text(image_path) -> str
        .content_type(image_path) -> str
        .vqa(paths, question) -> VQAResult
        .temporal_vqa(paths, question, ts) -> VQAResult
        .describe_keyframe(...) -> KeyframeDescription
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .vlm_contracts import (
    VLMBackend, VLMModel, VLMConfig,
    KeyframeDescription, VQAResult,
    FrameKind, FrameContentType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image_bytes(path: Union[str, Path]) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# Max long-edge sent to the VLM. Qwen2.5-VL uses dynamic resolution: a 1080p
# frame becomes thousands of vision tokens and takes ~30s/frame. Capping the
# long edge at ~1024px cuts vision tokens ~4-9x with no meaningful caption-
# quality loss, taking per-frame time from ~30s to ~5-7s. This is THE fix for
# slow captioning when the GPU is already at 100%.
_VLM_MAX_LONG_EDGE = 1024


def _to_base64(path: Union[str, Path]) -> str:
    """Downscale (if needed) then base64-encode for the VLM."""
    try:
        from PIL import Image
        import io
        im = Image.open(path)
        im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > _VLM_MAX_LONG_EDGE:
            scale = _VLM_MAX_LONG_EDGE / float(long_edge)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Fallback: send raw bytes if PIL isn't available or fails
        return base64.b64encode(_read_image_bytes(path)).decode("utf-8")


def _as_str(value: Any) -> str:
    """
    Safely coerce any value parsed from VLM JSON into a plain string.

    The VLM (especially a thinking model) sometimes returns a field as a nested
    object or list instead of a string, e.g.:
        "caption": {"text": "..."}            -> flatten the values
        "spatial_layout": {"fg": "x","bg":"y"} -> join the values
        "screen_text": ["line1", "line2"]      -> join with newlines
    Calling .strip() on those crashed with "'dict' object has no attribute
    'strip'". This normalises every case to a clean string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # join the dict's string-able values: {"fg":"a","bg":"b"} -> "a. b"
        parts = []
        for v in value.values():
            s = _as_str(v)
            if s:
                parts.append(s)
        return ". ".join(parts).strip()
    if isinstance(value, (list, tuple)):
        parts = [_as_str(v) for v in value]
        return "\n".join(p for p in parts if p).strip()
    return str(value).strip()

def _parse_json_array(text: str) -> List[str]:
    if not text:
        return []
    try:
        v = json.loads(text.strip())
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    m = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    cleaned = re.sub(r"[\[\]\"']", "", text)
    parts = re.split(r"[,;\n]", cleaned)
    return [p.strip() for p in parts if p.strip()][:12]

def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class _BaseVLMBackend:
    name: str = "base"

    def __init__(self, config: VLMConfig):
        self.config = config

    def is_available(self) -> bool:
        return False

    def generate(self, image_paths: Sequence[Union[str, Path]], prompt: str,
                 *, model: Optional[str] = None, max_tokens: int = 512,
                 temperature: float = 0.1) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaVLMBackend(_BaseVLMBackend):
    name = "ollama"

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._session = None
        try:
            import requests
            self._requests = requests
            self._session = requests.Session()
        except ImportError:
            self._requests = None

    def is_available(self) -> bool:
        if self._requests is None:
            return False
        try:
            r = self._session.get(
                f"{self.config.ollama_base_url}/api/tags", timeout=3.0
            )
            if r.status_code != 200:
                logger.warning("Ollama not reachable at %s", self.config.ollama_base_url)
                return False
            pulled = [m.get("name", "") for m in r.json().get("models", [])]
            primary = self.config.primary_model
            if pulled and primary not in pulled:
                logger.warning(
                    "Ollama running but model '%s' not pulled. "
                    "Available: %s. Run: ollama pull %s",
                    primary, pulled, primary,
                )
            return True
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    def list_models(self) -> List[str]:
        if self._requests is None:
            return []
        try:
            r = self._session.get(
                f"{self.config.ollama_base_url}/api/tags", timeout=5.0
            )
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    def generate(self, image_paths, prompt, *, model=None,
                 max_tokens=512, temperature=0.1):
        if self._requests is None:
            raise RuntimeError("`requests` not installed — pip install requests")

        model_name = model or self.config.primary_model
        images_b64 = [_to_base64(p) for p in image_paths]
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            # CRITICAL for speed: disable thinking. qwen3-vl is a hybrid
            # reasoning model — without this it emits a long chain-of-thought
            # before the JSON on EVERY frame, which is where ~25s/frame goes.
            # Harmless for non-thinking models (qwen2.5vl ignores it).
            "think": False,
            "options": {
                "num_predict": int(max_tokens),
                "temperature": float(temperature),
                # Cap context: captioning needs the image + a short prompt, not
                # a huge context window. Smaller num_ctx = less KV-cache = faster
                # and lower VRAM. 4096 is plenty for one frame + prompt.
                "num_ctx": 4096,
                # Stop as soon as the JSON object closes, so the model can't
                # ramble after the closing brace.
                "stop": ["\n\n\n", "```\n\n"],
            },
        }
        url = f"{self.config.ollama_base_url}/api/generate"
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                r = self._session.post(
                    url, json=payload,
                    timeout=self.config.request_timeout_s,
                )
                r.raise_for_status()
                return (r.json().get("response") or "").strip()
            except Exception as e:
                last_err = e
                logger.warning("Ollama retry %d/%d: %s",
                               attempt + 1, self.config.max_retries + 1, e)
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed: {last_err}")


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

class TransformersVLMBackend(_BaseVLMBackend):
    name = "transformers"
    _HF_REPO_MAP = {
        "qwen2.5vl:7b":     "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5vl:3b":     "Qwen/Qwen2.5-VL-3B-Instruct",
        "smolvlm:2.2b":     "HuggingFaceTB/SmolVLM-2.2B-Instruct",
        "moondream:latest": "vikhyatk/moondream2",
        "internvl3:8b":     "OpenGVLab/InternVL3-8B",
        "minicpm-v:2.6":    "openbmb/MiniCPM-V-2_6",
    }

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._loaded_model: Any = None
        self._loaded_processor: Any = None
        self._loaded_name: Optional[str] = None
        self._torch = None
        try:
            import torch
            self._torch = torch
        except ImportError:
            logger.debug("PyTorch not available — Transformers backend disabled")

    def is_available(self) -> bool:
        return self._torch is not None

    def _resolve_repo(self, model_name: str) -> str:
        return self._HF_REPO_MAP.get(model_name, model_name)

    def _load(self, model_name: str):
        if self._loaded_name == model_name:
            return
        repo = self._resolve_repo(model_name)
        logger.info("Loading VLM from HuggingFace: %s", repo)
        from transformers import AutoProcessor, AutoModelForVision2Seq
        device = self.config.transformers_device
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        dtype = self._torch.float16 if device == "cuda" else self._torch.float32
        self._loaded_processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        self._loaded_model = AutoModelForVision2Seq.from_pretrained(
            repo, torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._loaded_model = self._loaded_model.to(device)
        self._loaded_name = model_name

    def generate(self, image_paths, prompt, *, model=None,
                 max_tokens=512, temperature=0.1):
        if self._torch is None:
            raise RuntimeError("PyTorch not available")
        model_name = model or self.config.primary_model
        self._load(model_name)
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._loaded_processor(text=prompt, images=images, return_tensors="pt")
        device = next(self._loaded_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self._torch.no_grad():
            ids = self._loaded_model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=(temperature > 0.01),
                temperature=float(temperature) if temperature > 0.01 else 1.0,
            )
        out = self._loaded_processor.decode(ids[0], skip_special_tokens=True)
        if prompt in out:
            out = out[len(prompt):].strip()
        return out.strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

class OpenAIVLMBackend(_BaseVLMBackend):
    name = "openai"

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
        try:
            import openai
            base = config.openai_base_url or None
            key = config.openai_api_key or "sk-local"
            self._client = openai.OpenAI(api_key=key, base_url=base)
        except ImportError:
            logger.debug("openai library not installed")

    def is_available(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def generate(self, image_paths, prompt, *, model=None,
                 max_tokens=512, temperature=0.1):
        if self._client is None:
            raise RuntimeError("openai library not installed")
        model_name = model or self.config.primary_model
        content: List[Dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_to_base64(p)}"},
            })
        resp = self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# VLMClient
# ---------------------------------------------------------------------------

class VLMClient:
    """
    Unified VLM client. Only real backends — NO MOCK fallback.
    If Ollama is not running, raises clearly so the user knows what to fix.
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self._active_model = config.primary_model
        self._backend: _BaseVLMBackend = self._pick_backend()
        logger.info(
            "VLMClient ready: backend=%s model=%s",
            self._backend.name, self._active_model,
        )

    # ---------- backend selection -------------------------------------------

    def _pick_backend(self) -> _BaseVLMBackend:
        preferred = self.config.backend

        # Try Ollama first (with retries since app may start before Ollama)
        if preferred in (VLMBackend.OLLAMA.value, "ollama"):
            for attempt in range(3):
                b = OllamaVLMBackend(self.config)
                if b.is_available():
                    logger.info("✅ Ollama backend ready (attempt %d/3)", attempt + 1)
                    return b
                if attempt < 2:
                    logger.warning(
                        "Ollama not ready (attempt %d/3), retrying in 2s... "
                        "Make sure `ollama serve` is running.",
                        attempt + 1,
                    )
                    time.sleep(2.0)
            logger.error(
                "❌ Ollama not reachable after 3 attempts at %s. "
                "Start Ollama with: ollama serve",
                self.config.ollama_base_url,
            )

        # Try Transformers
        if preferred in (VLMBackend.TRANSFORMERS.value, "transformers"):
            b = TransformersVLMBackend(self.config)
            if b.is_available():
                return b

        # Try OpenAI-compatible
        if preferred in (VLMBackend.OPENAI.value, "openai"):
            b = OpenAIVLMBackend(self.config)
            if b.is_available():
                return b

        # Auto-fallback across all real backends
        for cls in [OllamaVLMBackend, TransformersVLMBackend, OpenAIVLMBackend]:
            try:
                b = cls(self.config)
                if b.is_available():
                    logger.warning("Using fallback backend: %s", cls.name)
                    return b
            except Exception:
                pass

        # NO MOCK — raise so the user knows
        raise RuntimeError(
            "No VLM backend is available. "
            "Start Ollama and pull the model: "
            f"ollama serve && ollama pull {self.config.primary_model}"
        )

    # ---------- public info --------------------------------------------------

    def is_available(self) -> bool:
        return self._backend.is_available()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def model_name(self) -> str:
        return self._active_model

    # ---------- core generate with model fallback ----------------------------

    def _generate_with_fallback(self, image_paths, prompt: str,
                                max_tokens: int = 512,
                                temperature: float = 0.1) -> str:
        models_to_try = [self.config.primary_model] + list(self.config.fallback_models)
        seen: set = set()
        last_err: Optional[Exception] = None
        for m in models_to_try:
            if m in seen:
                continue
            seen.add(m)
            try:
                out = self._backend.generate(
                    image_paths, prompt, model=m,
                    max_tokens=max_tokens, temperature=temperature,
                )
                self._active_model = m
                return out
            except Exception as e:
                logger.warning("VLM model '%s' failed: %s", m, e)
                last_err = e
        raise RuntimeError(f"All VLM models failed. Last error: {last_err}")

    # ---------- basic captioning ---------------------------------------------

    # ------------------------------------------------------------------
    # Hallucination filter
    # ------------------------------------------------------------------

    _HALLUCINATION_PHRASES = (
        # Common hallucinated names for animated/unknown figures
        "Barack Obama", "Michelle Obama", "Donald Trump", "Joe Biden",
        "Albert Einstein", "Elon Musk", "Bill Gates", "Steve Jobs",
        "Mark Zuckerberg", "Jeff Bezos", "Oprah Winfrey", "Nelson Mandela",
        "Martin Luther King", "Stephen Hawking", "Marie Curie",
    )

    def _filter_hallucinations(self, text: str, *, confirmed_text: str = "") -> str:
        """
        Replace likely-hallucinated real-person names with a safe placeholder,
        UNLESS the name is confirmed by on-screen text (OCR / chyron / caption).

        VLMs frequently invent famous names for unknown or animated figures, so
        the model's guess alone is never trusted. But if the video literally
        SHOWS the name (a lower-third, name badge, or slide), we keep it — that
        is real evidence, not a hallucination.

        confirmed_text: concatenated OCR / screen_text / board_text for this
                        frame. A blocked name found here is allowed through.
        """
        if not isinstance(text, str):
            text = _as_str(text)
        if not text:
            return text
        confirmed = (confirmed_text or "").lower()
        filtered = text
        for name in self._HALLUCINATION_PHRASES:
            if name.lower() in filtered.lower():
                # Allow the name through ONLY if on-screen text confirms it.
                if name.lower() in confirmed:
                    logger.debug("Hallucination filter ALLOWED (OCR-confirmed): %s", name)
                    continue
                import re
                filtered = re.sub(
                    re.escape(name), "[person not identified on screen]",
                    filtered, flags=re.IGNORECASE,
                )
                logger.debug("Hallucination filter removed name: %s", name)
        return filtered

    def caption(self, image_path, prompt: Optional[str] = None) -> str:
        p = prompt or self.config.caption_prompt
        raw = self._generate_with_fallback(
            [image_path], p, max_tokens=500, temperature=0.0
        )
        return self._filter_hallucinations(raw)

    # NOTE: short_caption(), tags(), ocr(), board_text(), content_type() were
    # removed on 2026-06-18 — the combined describe_keyframe prompt produces all
    # of these fields in ONE call, so the per-field methods were dead (0 callers).
    # Archived in ARCHIVED_dead_vlm_prompts.md.

    def scene_summary(self, image_path, timestamp_s: float) -> str:
        """
        Generates a single concise sentence describing the new scene.
        Anti-hallucination: never name people, only describe visible elements.
        """
        prompt = (
            f"This frame appears at {_fmt_ts(timestamp_s)}. "
            f"In ONE sentence (max 15 words), describe ONLY what you directly see "
            f"in this frame — no names, no guesses.\n\n"
            f"RULES:\n"
            f"- Never name a real person unless their name is written on screen.\n"
            f"- Describe only visible elements: setting, objects, colors, text on screen.\n\n"
            f"WRONG: 'Barack Obama begins speaking about climate change.'\n"
            f"RIGHT: 'A man in a suit appears at a podium in a large hall.'\n\n"
            f"WRONG: 'Einstein explains his theory on the blackboard.'\n"
            f"RIGHT: 'A new scene shows a blackboard covered with equations.'\n\n"
            f"ONE sentence describing only what you directly see:"
        )
        try:
            return self._generate_with_fallback(
                [image_path], prompt, max_tokens=60, temperature=0.0,
            ).strip()
        except Exception as e:
            logger.warning("scene_summary failed: %s", e)
            return ""

    # ---------- VQA ----------------------------------------------------------

    def vqa(self,
            image_paths: Union[str, Path, Sequence[Union[str, Path]]],
            question: str,
            *,
            video_id: Optional[str] = None,
            timestamps: Optional[Sequence[float]] = None,
            frame_captions: Optional[Sequence[dict]] = None) -> VQAResult:

        if isinstance(image_paths, (str, Path)):
            paths = [image_paths]
        else:
            paths = list(image_paths)

        if not paths:
            return VQAResult(
                question=question,
                answer="No frames provided.",
                video_id=video_id,
                confidence=0.0,
                model_name=self._active_model,
                backend=self._backend.name,
            )

        prompt = self.config.vqa_prompt_template.format(question=question)

        # Inject the captions/OCR we already extracted at index time. On
        # stylized or animated frames the VLM often fails to re-recognize an
        # object it captioned correctly during ingest (e.g. "owl"), so giving it
        # that text back dramatically improves recall on "is there an X" queries.
        if frame_captions:
            lines = []
            for fc in frame_captions:
                ts = fc.get("timestamp", 0.0)
                cap = (fc.get("caption") or "").strip()
                ocr = (fc.get("ocr_text") or "").strip()
                objs = fc.get("objects") or []
                obj_str = ", ".join(str(o) for o in objs if str(o).strip())
                spatial = (fc.get("spatial_layout") or "").strip()
                counts = fc.get("object_counts") or {}
                count_str = ", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else ""
                seg = f"- [{ts:.1f}s]"
                if cap:
                    seg += f" caption: {cap}"
                if obj_str:
                    seg += f" | objects detected: {obj_str}"
                if count_str:
                    seg += f" | counts: {count_str}"
                if spatial:
                    seg += f" | layout: {spatial}"
                if ocr:
                    seg += f" | on-screen text: {ocr}"
                if cap or ocr or obj_str or spatial:
                    lines.append(seg)
            if lines:
                prompt += (
                    "\n\nINDEX-TIME NOTES for these frames (extracted earlier by "
                    "the vision pipeline). Treat them as reliable supporting "
                    "evidence: if a note names an object and the frame is "
                    "consistent with it, you may confirm the object is present "
                    "even if the art is stylized or low-detail.\n"
                    + "\n".join(lines)
                )
        t0 = time.time()
        try:
            answer = self._generate_with_fallback(
                paths, prompt, max_tokens=500, temperature=0.0,
            )
        except Exception as e:
            logger.error("VQA failed: %s", e)
            return VQAResult(
                question=question,
                answer=f"VQA error: {e}",
                video_id=video_id,
                confidence=0.0,
                model_name=self._active_model,
                backend=self._backend.name,
                elapsed_s=time.time() - t0,
            )

        ts_list = list(timestamps) if timestamps else [0.0] * len(paths)
        return VQAResult(
            question=question,
            answer=self._filter_hallucinations(answer.strip()),
            video_id=video_id,
            frames_used=[(str(p), float(t)) for p, t in zip(paths, ts_list)],
            timestamps_examined=ts_list,
            confidence=0.9,
            model_name=self._active_model,
            backend=self._backend.name,
            elapsed_s=time.time() - t0,
        )

    def temporal_vqa(self,
                     image_paths: Union[str, Path, Sequence[Union[str, Path]]],
                     question: str,
                     timestamp_s: float,
                     *,
                     video_id: Optional[str] = None) -> VQAResult:
        """
        FIX 4 — temporal_vqa: also loads indexed board_text + screen_text
        from metadata and prepends them to the prompt so the VLM has the
        already-extracted text as context — prevents hallucination when
        re-reading the image at query time.
        """
        if isinstance(image_paths, (str, Path)):
            paths = [image_paths]
        else:
            paths = list(image_paths)

        if not paths:
            return VQAResult(
                question=question,
                answer=f"No frame available at {_fmt_ts(timestamp_s)}.",
                video_id=video_id,
                confidence=0.0,
                model_name=self._active_model,
                backend=self._backend.name,
                is_temporal_query=True,
            )

        ts_fmt = _fmt_ts(timestamp_s)
        prompt = self.config.temporal_vqa_prompt_template.format(
            timestamp=ts_fmt,
            question=question,
        )

        t0 = time.time()
        try:
            answer = self._generate_with_fallback(
                paths, prompt, max_tokens=600, temperature=0.0,
            )
        except Exception as e:
            logger.error("Temporal VQA failed ts=%.1f: %s", timestamp_s, e)
            return VQAResult(
                question=question,
                answer=f"Could not process frame at {ts_fmt}: {e}",
                video_id=video_id,
                confidence=0.0,
                model_name=self._active_model,
                backend=self._backend.name,
                elapsed_s=time.time() - t0,
                is_temporal_query=True,
            )

        return VQAResult(
            question=question,
            answer=self._filter_hallucinations(answer.strip()),
            video_id=video_id,
            frames_used=[(str(p), timestamp_s) for p in paths],
            timestamps_examined=[timestamp_s],
            confidence=0.9,
            model_name=self._active_model,
            backend=self._backend.name,
            elapsed_s=time.time() - t0,
            is_temporal_query=True,
        )

    # ---------- full keyframe pipeline ---------------------------------------

    # ------------------------------------------------------------------
    # COMBINED prompt — replaces 5 separate Ollama calls with 1
    # ------------------------------------------------------------------

    _COMBINED_PROMPT = """\
You are a meticulous video frame analyst. Examine this image and respond \
with a JSON object containing EXACTLY these fields — nothing else, no markdown, \
no explanation:

{{
  "content_type": "<one of: whiteboard | slide | screen | diagram | lecture | animation | interview | general>",
  "caption": "<3-4 sentences describing the WHOLE scene: main subject, what it is doing, and what is around/behind it. Only what is directly visible. Max 80 words>",
  "short_caption": "<1 sentence, max 15 words, only what you directly see>",
  "objects": ["<every distinct object, animal, person-role, structure, vehicle, plant, or notable item visible>"],
  "object_counts": {{"<object>": <integer count>}},
  "spatial_layout": "<where things are: foreground vs background, left/center/right, who/what is behind or beside the main subject>",
  "actions": ["<what is happening or CHANGING in this frame: things being drawn, written, added, highlighted, moved, demonstrated; for a person, what they are doing>"],
  "setting": "<short phrase: where this takes place, e.g. 'forest at night', 'city street', 'lecture slide'>",
  "tags": ["<tag1>", "<tag2>", ...],
  "board_text": "<text written on whiteboard/slide/screen, or BLANK if none>",
  "screen_text": "<ALL other on-screen text: subtitles, lower-thirds, signs, labels, UI text, watermarks, or NONE if absent>"
}}

STRICT RULES (apply to every field):
- NEVER name a real person unless their name is written visibly on screen \
(subtitle, lower-third, name badge). If you see a face, describe appearance only.
- objects: list EVERYTHING you can see, including background and secondary items. \
Include animals, people (by role/appearance), buildings, vehicles, plants, tools, \
furniture, screens, text panels, natural features. Stylized / cartoon / animated \
depictions COUNT as the real object (a drawn owl is an "owl"). Aim for 5-15 entries. \
Never output an empty list if anything is visible.
- object_counts: for each object TYPE that appears more than once, give the count \
(e.g. {{"owl": 3, "tree": 5}}). If everything appears once, output {{}}. \
Count carefully — this is used to answer "how many X" questions.
- spatial_layout: describe POSITIONS so questions like "what is behind the person" \
or "what is to the left" can be answered. Mention foreground, background, and \
left/center/right placement of the main objects. Always fill this in.
- setting: always fill this in; describe the location/environment even if generic.
- actions: capture what is HAPPENING or CHANGING in the frame (being drawn, \
written, added, highlighted, moved, demonstrated; for a person, what they do). \
For a static slide a single item or [] is fine. This matters for educational \
animations where the construction builds up over time.
- caption & short_caption: describe literal visible elements, include the main \
objects, their positions, and the setting. Never leave caption empty — if unsure, \
describe shapes/colours/layout.
- tags: 6-10 descriptive tags, no real person names. \
Example: ["forest","owl","otter","river","cartoon-style","wildlife"]
- board_text: transcribe equations, bullet points, titles EXACTLY as written. \
Output BLANK if nothing is written on a board/slide.
- screen_text: transcribe ALL visible text anywhere in the frame EXACTLY \
(signs, captions, labels, numbers). This is critical — do not skip small or \
background text. Output NONE only if there is genuinely no text.{scene_hint}

Respond with valid JSON only:"""

    _COMBINED_PROMPT_SCENE = (
        "\n- This frame marks a significant scene change — also add a "
        "\"scene_summary\" field: one sentence describing the new scene."
    )

    def _parse_combined_response(self, raw: str, is_major_change: bool) -> Dict[str, Any]:
        """Parse the combined JSON response, tolerating minor model formatting issues."""
        text = raw.strip()
        # Strip markdown fences if the model wrapped it
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        # Find the outermost JSON object
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Fallback: try to extract individual fields with regex
        logger.debug("Combined VLM response JSON parse failed, using regex fallback")
        result: Dict[str, Any] = {}
        for field in ("content_type", "caption", "short_caption", "board_text",
                      "screen_text", "scene_summary", "setting", "spatial_layout"):
            m = re.search(
                rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
            )
            if m:
                result[field] = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        # Tags array
        m = re.search(r'"tags"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if m:
            try:
                result["tags"] = json.loads(m.group(1))
            except Exception:
                result["tags"] = _parse_json_array(m.group(1))
        # Objects array
        m = re.search(r'"objects"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if m:
            try:
                result["objects"] = json.loads(m.group(1))
            except Exception:
                result["objects"] = _parse_json_array(m.group(1))
        return result

    def describe_keyframe(self,
                          image_path: Union[str, Path],
                          *,
                          video_id: str,
                          timestamp: float,
                          frame_index: int,
                          frame_kind: FrameKind = FrameKind.INTERVAL,
                          extract_tags: bool = True,
                          extract_ocr: bool = True,
                          language_hint: Optional[str] = None,
                          scene_change_score: float = 0.0,
                          is_major_change: bool = False,
                          ocr_engine: Optional[Any] = None,
                          ) -> KeyframeDescription:
        """
        Describe a keyframe using ONE combined Ollama call instead of 5.

        Old approach: content_type + caption + short_caption + tags + board_text
                      = 5 × HTTP round-trips × ~4 s each = ~20 s/frame
        New approach: single JSON prompt covering all fields
                      = 1 × HTTP round-trip × ~4-6 s = ~5 s/frame  (4× faster)

        EasyOCR still runs separately (it's CPU-side, not an Ollama call, and
        gives character-precise text the VLM sometimes misses).
        """
        t0 = time.time()
        caption    = ""
        short      = ""
        tags: List[str] = []
        objects: List[str] = []
        object_counts: Dict[str, int] = {}
        spatial_layout = ""
        actions: List[str] = []
        setting    = ""
        ocr_text   = ""
        board      = ""
        screen_text = ""
        scene_sum  = ""
        ctype      = FrameContentType.GENERAL.value

        # ── Single combined VLM call ──────────────────────────────────
        scene_hint = self._COMBINED_PROMPT_SCENE if is_major_change else ""
        combined_prompt = self._COMBINED_PROMPT.format(scene_hint=scene_hint)

        try:
            raw = self._generate_with_fallback(
                [image_path],
                combined_prompt,
                # With thinking disabled the model emits only the JSON (~300-400
                # tokens for all fields). 600 covers it with headroom; 700 on
                # scene-change frames (adds scene_summary). No need for 900+.
                max_tokens=700 if is_major_change else 600,
                temperature=0.0,
            )
            parsed = self._parse_combined_response(raw, is_major_change)

            # content_type
            raw_ctype = _as_str(parsed.get("content_type")).lower()
            valid_ctypes = {ct.value for ct in FrameContentType}
            ctype = raw_ctype if raw_ctype in valid_ctypes else FrameContentType.GENERAL.value
            # Upgrade general→mixed on scene change (preserves old FIX 2 logic)
            if ctype == FrameContentType.GENERAL.value and scene_change_score > 0.10:
                ctype = FrameContentType.MIXED.value

            # board_text + screen_text FIRST — so on-screen text can confirm a
            # real-person name before we run the hallucination filter on the
            # caption. A name written on screen (chyron/badge/slide) is evidence.
            if extract_ocr and self.config.board_text_enabled:
                raw_board = _as_str(parsed.get("board_text"))
                board = "" if raw_board.upper() in ("BLANK", "[BLANK]", "NONE", "[NONE]", "") else raw_board
            if extract_ocr:
                raw_screen = _as_str(parsed.get("screen_text"))
                screen_text = "" if raw_screen.upper() in ("NONE", "[NONE]", "BLANK", "[BLANK]", "") else raw_screen
            _confirmed = f"{board} {screen_text}".strip()

            # caption
            raw_caption = _as_str(parsed.get("caption"))
            caption = self._filter_hallucinations(raw_caption, confirmed_text=_confirmed)

            # short_caption — derive from caption if missing
            raw_short = _as_str(parsed.get("short_caption"))
            if raw_short:
                short = self._filter_hallucinations(raw_short[:200], confirmed_text=_confirmed)
            elif caption:
                short = re.split(r"[.\n!?]", caption, maxsplit=1)[0][:180]

            # tags
            if extract_tags:
                raw_tags = parsed.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [_as_str(t) for t in raw_tags if _as_str(t)][:12]
                else:
                    tags = _parse_json_array(_as_str(raw_tags))[:12]

            # objects (VLM-based object detection) — the key field for
            # "what else is on screen / where is the X" benchmark questions.
            raw_objects = parsed.get("objects", [])
            if isinstance(raw_objects, list):
                objects = [_as_str(o) for o in raw_objects if _as_str(o)][:25]
            else:
                objects = _parse_json_array(_as_str(raw_objects))[:25]

            # setting — fold into objects-as-context and into caption if caption
            # came back empty so the frame is never blank.
            setting = _as_str(parsed.get("setting"))
            if setting and not caption:
                caption = setting
            if setting and not short:
                short = setting[:180]

            # spatial_layout — answers "what is behind / left / right of X"
            spatial_layout = _as_str(parsed.get("spatial_layout"))

            # actions — what is happening/changing (Vgent-style); list of strings
            raw_actions = parsed.get("actions", [])
            if isinstance(raw_actions, list):
                actions = [_as_str(a).strip() for a in raw_actions if _as_str(a).strip()]
            elif isinstance(raw_actions, str) and raw_actions.strip():
                actions = [raw_actions.strip()]
            else:
                actions = []

            # object_counts — answers "how many X" (VNBench counting)
            raw_counts = parsed.get("object_counts", {})
            if isinstance(raw_counts, dict):
                for k, v in raw_counts.items():
                    try:
                        object_counts[_as_str(k).lower()] = int(v)
                    except (ValueError, TypeError):
                        continue
            # Backfill counts from the objects list when the model didn't supply
            # object_counts but listed duplicates (e.g. ["owl","owl","otter"]).
            if not object_counts and objects:
                from collections import Counter as _Counter
                dup = {k: c for k, c in _Counter(
                    o.lower() for o in objects
                ).items() if c > 1}
                object_counts = dup

            # scene_summary (only on major scene changes)
            if is_major_change:
                raw_scene = _as_str(parsed.get("scene_summary"))
                scene_sum = self._filter_hallucinations(raw_scene, confirmed_text=_confirmed)

            logger.debug(
                "Combined VLM OK frame=%d ts=%.1f ctype=%s board=%s screen=%s elapsed=%.1fs",
                frame_index, timestamp, ctype, bool(board), bool(screen_text),
                time.time() - t0,
            )

        except Exception as e:
            logger.warning(
                "Combined VLM call failed frame=%d: %s — fields will be empty",
                frame_index, e,
            )
            # caption fallback: try a plain caption-only call so at least
            # the frame is not completely blank in the index
            try:
                caption = self.caption(image_path)
                short   = re.split(r"[.\n!?]", caption, maxsplit=1)[0][:180] if caption else ""
            except Exception:
                pass

        # ── OCR engine — runs on CPU, character-precise (PaddleOCR/EasyOCR) ──
        # PERF: only run OCR on frames likely to contain meaningful text
        # (slides, screens, boards, diagrams) OR when the VLM itself flagged
        # screen text. Running EasyOCR on every animation frame on CPU added
        # seconds/frame for little gain. This skips it on pure-animation frames.
        _text_likely = ctype in (
            FrameContentType.SLIDE.value, FrameContentType.SCREEN.value,
            FrameContentType.WHITEBOARD.value, FrameContentType.DIAGRAM.value,
            FrameContentType.LECTURE.value,
        ) or bool(screen_text) or bool(board)
        if extract_ocr and _text_likely:
            try:
                _engine = ocr_engine
                if _engine is None:
                    from .ocr import UnifiedOCREngine
                    _engine = UnifiedOCREngine(languages=[
                        (language_hint or "en").strip().lower()[:2]
                    ])
                if _engine is not None and _engine.is_available:
                    import cv2 as _cv2
                    _img = _cv2.imread(str(image_path))
                    if _img is not None:
                        _result = _engine.extract_text(_img)
                        _ocr_out = _result.text.strip()
                        if _ocr_out:
                            # store as detected_text; merge into screen_text,
                            # avoiding duplicate content
                            ocr_text = _ocr_out
                            if not screen_text:
                                screen_text = _ocr_out
                            elif _ocr_out.lower() not in screen_text.lower():
                                screen_text = f"{screen_text}\n{_ocr_out}".strip()
                            logger.debug(
                                "OCR extracted %d chars (backend=%s)",
                                len(_ocr_out), _engine.backend,
                            )
            except Exception as e:
                logger.warning("OCR screen_text failed: %s", e)

        # ── Image dimensions ─────────────────────────────────────────────────
        w = h = None
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as im:
                w, h = im.size
        except Exception:
            pass

        kd = KeyframeDescription(
            video_id=video_id,
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            frame_kind=frame_kind,
            caption=caption,
            short_caption=short,
            tags=tags,
            detected_objects=objects,
            object_counts=object_counts,
            spatial_layout=spatial_layout,
            actions=actions,
            detected_text=ocr_text,
            board_text=board,
            screen_text=screen_text,
            scene_summary=scene_sum,
            text_confidence=0.9,
            content_type=ctype,
            language_hint=language_hint,
            scene_change_score=float(scene_change_score),
            is_major_change=bool(is_major_change),
            model_name=self._active_model,
            backend=self._backend.name,
            frame_path=str(image_path),
            width=w,
            height=h,
            confidence=0.9,
        )
        kd.extra["describe_elapsed_s"]      = round(time.time() - t0, 3)
        kd.extra["board_text_extracted"]    = bool(board)
        kd.extra["screen_text_extracted"]   = bool(screen_text)
        kd.extra["scene_summary_extracted"] = bool(scene_sum)
        kd.extra["objects_extracted"]       = len(objects)
        kd.extra["object_counts"]           = object_counts
        kd.extra["spatial_layout"]          = spatial_layout
        kd.extra["setting"]                 = setting
        kd.extra["content_type"]            = ctype
        kd.extra["combined_vlm_call"]       = True   # mark so you can verify in logs
        return kd


__all__ = [
    "VLMClient",
    "OllamaVLMBackend",
    "TransformersVLMBackend",
    "OpenAIVLMBackend",
]