"""
core/clip_embedder.py
=====================

CLIP-style multimodal embedder for cross-modal search (text ↔ image).

Two backends, chosen automatically:
    1. sentence-transformers (preferred) — loads CLIP locally
    2. open_clip                            — alternative implementation
    (mock backend removed — only real CLIP backends are used)

Produces L2-normalised vectors of `embedding_dim` dimensions (default 512 for
ViT-B/32). Vectors live in the SAME shared embedding space for text and image,
so you can FAISS-search "a whiteboard with equations" against image embeddings
directly.

Public API:
    CLIPEmbedder(config=...)
        .embed_image(path_or_pil) -> np.ndarray[dim]
        .embed_images(paths) -> np.ndarray[N, dim]
        .embed_text(str or List[str]) -> np.ndarray[N, dim]
        .similarity(a, b) -> float (cosine)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# sentence-transformers CLIP
# ---------------------------------------------------------------------------


class _STClipBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading CLIP model '%s' on %s ...", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension() or 512

    def embed_image(self, path):
        from PIL import Image  # type: ignore
        img = Image.open(path).convert("RGB") if not hasattr(path, "mode") else path
        v = self._model.encode([img], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return v[0].astype("float32")

    def embed_images(self, paths):
        from PIL import Image  # type: ignore
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return self._model.encode(imgs, normalize_embeddings=True,
                                  convert_to_numpy=True,
                                  show_progress_bar=False).astype("float32")

    def embed_text(self, text):
        if isinstance(text, str):
            v = self._model.encode([text], normalize_embeddings=True,
                                   convert_to_numpy=True,
                                   show_progress_bar=False)
            return v[0].astype("float32")
        return self._model.encode(list(text), normalize_embeddings=True,
                                  convert_to_numpy=True,
                                  show_progress_bar=False).astype("float32")


# ---------------------------------------------------------------------------
# open_clip
# ---------------------------------------------------------------------------


class _OpenClipBackend:
    name = "open_clip"

    # Maps VLMConfig.clip_model names → (open_clip arch, pretrained weights)
    _MODEL_MAP = {
        "sentence-transformers/clip-ViT-B-32": ("ViT-B-32", "laion2b_s34b_b79k"),
        "ViT-B-32":  ("ViT-B-32",  "laion2b_s34b_b79k"),
        "ViT-B-16":  ("ViT-B-16",  "laion2b_s34b_b88k"),
        "ViT-L-14":  ("ViT-L-14",  "laion2b_s32b_b82k"),
        "ViT-H-14":  ("ViT-H-14",  "laion2b_s32b_b79k"),
    }

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32",
                 pretrained: str = "",
                 device: str = "auto"):
        import open_clip  # type: ignore
        import torch      # type: ignore

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._torch = torch

        arch, weights = self._MODEL_MAP.get(
            model_name, ("ViT-B-32", "laion2b_s34b_b79k")
        )
        if pretrained:
            weights = pretrained

        logger.info(
            "Loading open_clip model arch=%s weights=%s on %s",
            arch, weights, device,
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=weights
        )
        self._model.eval().to(device)
        self._tokenizer = open_clip.get_tokenizer(arch)
        self.dim = self._model.visual.output_dim  # type: ignore
        logger.info("open_clip ready (arch=%s, dim=%d)", arch, self.dim)

    def embed_image(self, path):
        from PIL import Image  # type: ignore
        img = Image.open(path).convert("RGB") if not hasattr(path, "mode") else path
        tens = self._preprocess(img).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            emb = self._model.encode_image(tens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].astype("float32")

    def embed_images(self, paths):
        from PIL import Image  # type: ignore
        ts = [self._preprocess(Image.open(p).convert("RGB")) for p in paths]
        batch = self._torch.stack(ts).to(self._device)
        with self._torch.no_grad():
            emb = self._model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def embed_text(self, text):
        items = [text] if isinstance(text, str) else list(text)
        tokens = self._tokenizer(items).to(self._device)
        with self._torch.no_grad():
            emb = self._model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        arr = emb.cpu().numpy().astype("float32")
        return arr[0] if isinstance(text, str) else arr


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CLIPEmbedder:
    """
    Multimodal embedder. Selects best available backend transparently.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/clip-ViT-B-32",
                 device: str = "auto",
                 embedding_dim: int = 512):
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self._backend = None

        # Try sentence-transformers
        try:
            self._backend = _STClipBackend(model_name, device=device)
            logger.info("CLIPEmbedder: sentence-transformers ready (dim=%d)",
                        self._backend.dim)
            return
        except Exception as e:
            logger.debug("sentence-transformers backend failed: %s", e)

        # Try open_clip
        try:
            self._backend = _OpenClipBackend(model_name=model_name, device=device)
            logger.info("CLIPEmbedder: open_clip ready (dim=%d)", self._backend.dim)
            return
        except Exception as e:
            logger.debug("open_clip backend failed: %s", e)

        # No real backend — raise clearly, never use mock
        raise RuntimeError(
            "No real CLIP backend available. "
            "You already have torch — install open_clip with:\n"
            "  pip install open-clip-torch Pillow\n"
            "Or sentence-transformers:\n"
            "  pip install sentence-transformers Pillow"
        )

    # ---------- info --------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def dim(self) -> int:
        return getattr(self._backend, "dim", self.embedding_dim)

    def is_real(self) -> bool:
        return True  # mock backend removed — if we got here, it's always real

    # ---------- embedding ---------------------------------------------------

    def embed_image(self, path_or_image):
        return self._backend.embed_image(path_or_image)

    def embed_images(self, paths: Sequence):
        if hasattr(self._backend, "embed_images"):
            return self._backend.embed_images(list(paths))
        if np is None:
            raise RuntimeError("NumPy required")
        return np.stack([self._backend.embed_image(p) for p in paths])

    def embed_text(self, text: Union[str, Sequence[str]]):
        return self._backend.embed_text(text)

    # ---------- similarity -------------------------------------------------

    def similarity(self, a, b) -> float:
        if np is None:
            raise RuntimeError("NumPy required")
        a = np.asarray(a, dtype="float32").flatten()
        b = np.asarray(b, dtype="float32").flatten()
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))


__all__ = ["CLIPEmbedder"]