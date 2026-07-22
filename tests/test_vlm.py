"""
tests/test_vlm.py
=================

Comprehensive test suite for the VLM subsystem.

Covers:
    - vlm_contracts: dataclass serialisation and invariants
    - vlm_client:    MockVLMBackend, model fallback, caption/short/tags/ocr/vqa
    - keyframe_extractor: extraction with synthetic video
    - clip_embedder:  mock + similarity
    - vision_indexer: end-to-end with mock backends
    - vqa_engine:     retrieve + answer with InMemoryFrameIndex

Run:
    pytest tests/test_vlm.py -v
    VLM_FORCE_MOCK=1 CLIP_FORCE_MOCK=1 pytest tests/test_vlm.py -v

The test suite is designed to pass WITHOUT Ollama/GPU by forcing mock backends.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

import pytest

# Force mock backends for deterministic CI
os.environ.setdefault("VLM_FORCE_MOCK", "1")
os.environ.setdefault("CLIP_FORCE_MOCK", "1")

# Make the project importable whether tests run from repo root or /tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vlm_contracts import (
    VLMConfig, VLMBackend, VLMModel, FrameKind,
    KeyframeDescription, VQAResult,
)
from core.vlm_client import (
    VLMClient, MockVLMBackend, _parse_json_array,
)
from core.clip_embedder import CLIPEmbedder
from core.vqa_engine import VQAEngine, InMemoryFrameIndex
from core.vision_indexer import VisionIndexer, VisualChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_image(path: Path, size=(128, 128), color=(180, 40, 40)) -> None:
    """Create a small JPG on disk. Falls back to writing a tiny valid file
    if Pillow is not available (VLMClient in mock mode does not decode)."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
        img = Image.new("RGB", size, color=color)
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255, 255, 255), width=3)
        img.save(path, "JPEG", quality=85)
    except ImportError:
        # Minimal JPEG magic bytes so os.path.exists and open() work
        path.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        )


def _make_dummy_video(path: Path, seconds: int = 5, fps: int = 10,
                      size=(160, 120)) -> bool:
    """Create a tiny synthetic video with distinct scenes. Returns False if OpenCV missing."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        return False

    total = seconds * fps
    n_scenes = max(2, seconds)
    for i in range(total):
        scene = i * n_scenes // total
        # Each scene gets a distinct colour + distinct rectangle
        base = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        base[:, :, scene % 3] = 60 + (scene * 40) % 195
        base[:, :, (scene + 1) % 3] = 40
        # Add a moving rectangle for more visual variation
        x = 10 + (i * 5) % (size[0] - 40)
        y = 10 + (scene * 15) % (size[1] - 40)
        cv2.rectangle(base, (x, y), (x + 30, y + 30),
                      (255, 255, 255), -1)
        cv2.putText(base, f"S{scene}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        writer.write(base)
    writer.release()
    return True


# ===========================================================================
# Contracts
# ===========================================================================


class TestContracts:

    def test_vlm_config_defaults(self):
        c = VLMConfig()
        assert c.enabled is True
        assert c.primary_model
        assert c.frame_interval_s > 0
        assert c.max_frames_per_video > 0
        assert c.clip_embedding_dim > 0

    def test_vlm_config_from_settings(self):
        class DummySettings:
            vision_model_enabled = False
            vlm_primary_model = "smolvlm:2.2b"
            vlm_backend = "mock"
            vision_frame_interval_s = 2.0
            vision_max_frames = 50
            vlm_clip_enabled = False
        c = VLMConfig.from_settings(DummySettings())
        assert c.enabled is False
        assert c.primary_model == "smolvlm:2.2b"
        assert c.backend == "mock"
        assert c.frame_interval_s == 2.0
        assert c.max_frames_per_video == 50
        assert c.clip_enabled is False

    def test_keyframe_description_serialisation(self):
        d = KeyframeDescription(
            video_id="v1", timestamp=12.34, frame_index=42,
            caption="A person at a whiteboard.",
            short_caption="Person at whiteboard.",
            tags=["person", "whiteboard"],
            detected_text="E = mc^2",
            model_name="mock", backend="mock",
        )
        # doc_id stable and short
        assert isinstance(d.doc_id, str)
        assert len(d.doc_id) == 16
        # searchable_text combines everything
        st = d.searchable_text
        assert "Person at whiteboard" in st
        assert "person whiteboard" in st
        assert "E = mc^2" in st
        # round-trip
        dct = d.to_dict()
        d2 = KeyframeDescription.from_dict(dct)
        assert d2.video_id == d.video_id
        assert d2.frame_index == d.frame_index
        assert d2.caption == d.caption

    def test_vqa_result_to_dict(self):
        r = VQAResult(question="What?", answer="something",
                      frames_used=[("/p", 1.0)], model_name="m", backend="mock")
        d = r.to_dict()
        assert d["question"] == "What?"
        assert d["answer"] == "something"
        assert d["frames_used"] == [("/p", 1.0)] or d["frames_used"] == [["/p", 1.0]]


# ===========================================================================
# VLM client (mock backend)
# ===========================================================================


class TestVLMClient:

    @pytest.fixture
    def tmp_image(self, tmp_path):
        p = tmp_path / "frame.jpg"
        _make_dummy_image(p)
        return p

    def test_client_picks_mock_when_forced(self):
        cfg = VLMConfig(backend="mock")
        c = VLMClient(cfg)
        assert c.backend_name == "mock"
        assert c.is_available() is True

    def test_caption(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        cap = c.caption(str(tmp_image))
        assert isinstance(cap, str)
        assert len(cap) > 0
        assert "frame" in cap.lower() or "scene" in cap.lower()

    def test_short_caption_is_short(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        sc = c.short_caption(str(tmp_image))
        assert isinstance(sc, str)
        assert len(sc) < 200

    def test_tags_returns_list(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        tags = c.tags(str(tmp_image))
        assert isinstance(tags, list)
        assert len(tags) >= 1
        assert all(isinstance(t, str) for t in tags)

    def test_ocr_returns_string(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        txt = c.ocr(str(tmp_image))
        assert isinstance(txt, str)

    def test_vqa_single_frame(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        r = c.vqa(str(tmp_image), "What's happening in this frame?",
                  video_id="v1", timestamps=[10.0])
        assert isinstance(r, VQAResult)
        assert r.question.startswith("What")
        assert r.answer
        assert r.video_id == "v1"
        assert len(r.frames_used) == 1

    def test_vqa_multi_frame(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"f{i}.jpg"
            _make_dummy_image(p)
            paths.append(str(p))
        c = VLMClient(VLMConfig(backend="mock"))
        r = c.vqa(paths, "Summarise these frames.", video_id="v2",
                  timestamps=[1.0, 2.0, 3.0])
        assert len(r.frames_used) == 3
        assert r.answer

    def test_describe_keyframe_full(self, tmp_image):
        c = VLMClient(VLMConfig(backend="mock"))
        kd = c.describe_keyframe(
            str(tmp_image),
            video_id="vid",
            timestamp=5.5,
            frame_index=10,
            frame_kind=FrameKind.SCENE_CHANGE,
            extract_tags=True,
            extract_ocr=True,
            language_hint="en",
        )
        assert isinstance(kd, KeyframeDescription)
        assert kd.video_id == "vid"
        assert kd.frame_index == 10
        assert kd.frame_kind == FrameKind.SCENE_CHANGE
        assert kd.caption
        assert len(kd.tags) > 0
        assert kd.backend == "mock"
        assert kd.frame_path == str(tmp_image)

    def test_model_fallback_on_failure(self, tmp_image, monkeypatch):
        """If primary model raises, fallback model should be tried."""
        cfg = VLMConfig(backend="mock",
                        primary_model="fake_primary",
                        fallback_models=["fake_fallback"])
        c = VLMClient(cfg)

        calls = {"n": 0}
        orig = c._backend.generate

        def flaky(image_paths, prompt, *, model=None, **kw):
            calls["n"] += 1
            if model == "fake_primary":
                raise RuntimeError("simulated primary failure")
            return orig(image_paths, prompt, model=model, **kw)

        monkeypatch.setattr(c._backend, "generate", flaky)

        out = c.caption(str(tmp_image))
        assert out
        # primary tried, then fallback
        assert calls["n"] >= 2
        assert c.model_name == "fake_fallback"


# ===========================================================================
# JSON-array parser
# ===========================================================================


class TestJsonArrayParser:

    def test_clean_array(self):
        assert _parse_json_array('["a", "b", "c"]') == ["a", "b", "c"]

    def test_dirty_prefix(self):
        s = 'Here are the tags: ["diagram", "text", "arrow"] — hope it helps.'
        out = _parse_json_array(s)
        assert "diagram" in out
        assert "text" in out
        assert "arrow" in out

    def test_comma_fallback(self):
        out = _parse_json_array("diagram, text, arrow")
        assert out == ["diagram", "text", "arrow"]

    def test_empty(self):
        assert _parse_json_array("") == []


# ===========================================================================
# CLIP embedder (mock)
# ===========================================================================


class TestCLIPEmbedder:

    def test_mock_embed_dim(self):
        c = CLIPEmbedder(force_mock=True, embedding_dim=256)
        v = c.embed_text("hello world")
        assert v.shape == (256,)

    def test_text_image_same_space(self, tmp_path):
        p = tmp_path / "img.jpg"
        _make_dummy_image(p)
        c = CLIPEmbedder(force_mock=True, embedding_dim=128)
        t_vec = c.embed_text("red square")
        i_vec = c.embed_image(str(p))
        assert t_vec.shape == i_vec.shape == (128,)
        # cosine similarity must be finite
        s = c.similarity(t_vec, i_vec)
        assert -1.0 <= s <= 1.0

    def test_batch_text(self):
        c = CLIPEmbedder(force_mock=True, embedding_dim=64)
        arr = c.embed_text(["a", "b", "c"])
        assert arr.shape == (3, 64)

    def test_deterministic_mock(self):
        c = CLIPEmbedder(force_mock=True, embedding_dim=32)
        v1 = c.embed_text("same text")
        v2 = c.embed_text("same text")
        # deterministic hash-based mock
        assert (v1 == v2).all()


# ===========================================================================
# Keyframe extractor
# ===========================================================================


@pytest.mark.skipif(shutil.which is None, reason="shutil missing (impossible)")
class TestKeyframeExtractor:

    def test_extraction_on_synthetic_video(self, tmp_path):
        try:
            import cv2  # noqa
        except ImportError:
            pytest.skip("OpenCV not installed")

        video = tmp_path / "tiny.mp4"
        ok = _make_dummy_video(video, seconds=4, fps=10)
        if not ok:
            pytest.skip("OpenCV video writer unavailable")

        from core.keyframe_extractor import KeyframeExtractor
        kex = KeyframeExtractor(
            output_dir=str(tmp_path / "frames"),
            interval_s=1.0,
            max_frames=20,
            scene_threshold=0.25,
            dedup_phash_distance=0,   # disable phash dedup on monochrome synth
        )
        frames = kex.extract(str(video), video_id="testvid")
        assert len(frames) >= 2  # at least a couple of keyframes
        for f in frames:
            assert os.path.exists(f.path)
            assert f.width > 0 and f.height > 0
            assert f.timestamp_s >= 0


# ===========================================================================
# Vision indexer (end-to-end with mocks)
# ===========================================================================


class TestVisionIndexer:

    def test_init_with_mock(self, tmp_path):
        cfg = VLMConfig(backend="mock",
                        clip_enabled=True,
                        ocr_fallback_enabled=False)
        vi = VisionIndexer(
            vlm_config=cfg,
            frames_dir=str(tmp_path / "frames"),
            descriptions_dir=str(tmp_path / "desc"),
        )
        assert vi.vlm.backend_name == "mock"
        assert vi.clip is not None
        assert vi.clip.backend_name == "mock"

    def test_index_video_full_pipeline(self, tmp_path):
        try:
            import cv2  # noqa
        except ImportError:
            pytest.skip("OpenCV not installed")

        video = tmp_path / "v.mp4"
        if not _make_dummy_video(video, seconds=4, fps=10):
            pytest.skip("OpenCV video writer unavailable")

        cfg = VLMConfig(
            backend="mock",
            clip_enabled=True,
            ocr_fallback_enabled=False,
            frame_interval_s=1.0,
            max_frames_per_video=10,
            clip_embedding_dim=128,
        )
        vi = VisionIndexer(
            vlm_config=cfg,
            frames_dir=str(tmp_path / "frames"),
            descriptions_dir=str(tmp_path / "desc"),
        )
        chunks = vi.index_video(
            str(video),
            video_id="vtest",
            language_hint="en",
            extract_tags=True,
            extract_ocr=True,
            save_json=True,
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, VisualChunk)
            assert c.video_id == "vtest"
            assert c.text  # non-empty
            assert c.embedding is not None
            assert c.embedding_dim == 128
            assert c.metadata.get("frame_path")

        # JSON persisted
        json_file = tmp_path / "desc" / "vtest_vision.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["video_id"] == "vtest"
        assert len(data["descriptions"]) == len(chunks)

    def test_disabled(self, tmp_path):
        cfg = VLMConfig(enabled=False, backend="mock")
        vi = VisionIndexer(
            vlm_config=cfg,
            frames_dir=str(tmp_path / "frames"),
            descriptions_dir=str(tmp_path / "desc"),
        )
        out = vi.index_video("/no/such/file.mp4", video_id="x")
        assert out == []


# ===========================================================================
# VQA engine
# ===========================================================================


class TestVQAEngine:

    def _build(self, tmp_path, n_frames=5):
        # Build synthetic visual chunks
        clip = CLIPEmbedder(force_mock=True, embedding_dim=64)
        chunks = []
        for i in range(n_frames):
            p = tmp_path / f"f{i}.jpg"
            _make_dummy_image(p)
            emb = clip.embed_image(str(p)).tolist()
            chunks.append(VisualChunk(
                chunk_id=f"c{i}",
                video_id="vq",
                text=f"Frame {i} caption about topic {i % 2}",
                timestamp=float(i),
                end_timestamp=float(i) + 1.0,
                embedding=emb,
                embedding_dim=64,
                metadata={
                    "frame_path": str(p),
                    "caption": f"Caption {i}",
                    "tags": [f"tag{i}"],
                    "ocr_text": "",
                },
            ))
        idx = InMemoryFrameIndex()
        idx.add_chunks(chunks)
        vlm = VLMClient(VLMConfig(backend="mock"))
        engine = VQAEngine(vlm=vlm, clip=clip, frame_index=idx,
                           top_k_retrieve=5, top_n_answer=2)
        return engine, idx

    def test_retrieve(self, tmp_path):
        engine, idx = self._build(tmp_path, n_frames=5)
        assert len(idx) == 5
        res = engine.retrieve("topic 0", top_k=3)
        assert 1 <= len(res) <= 3
        for r in res:
            assert "score" in r
            assert "frame_path" in r

    def test_answer_returns_dict(self, tmp_path):
        engine, _ = self._build(tmp_path, n_frames=4)
        out = engine.answer("What is visible in frame 2?",
                            video_id="vq", top_n=2,
                            include_retrieval=True)
        assert isinstance(out, dict)
        assert "answer" in out
        assert "frames_used" in out
        assert out["backend"] == "mock"
        assert out["elapsed_s"] >= 0

    def test_answer_with_empty_index(self, tmp_path):
        clip = CLIPEmbedder(force_mock=True, embedding_dim=64)
        idx = InMemoryFrameIndex()
        vlm = VLMClient(VLMConfig(backend="mock"))
        engine = VQAEngine(vlm=vlm, clip=clip, frame_index=idx)
        out = engine.answer("anything?")
        assert out["answer"]
        assert out["frames"] == []

    def test_video_scoped_retrieval(self, tmp_path):
        engine, idx = self._build(tmp_path, n_frames=3)
        # Add frames for a 2nd video
        clip = CLIPEmbedder(force_mock=True, embedding_dim=64)
        for i in range(2):
            p = tmp_path / f"g{i}.jpg"
            _make_dummy_image(p)
            idx._frames.append(type(idx._frames[0])(
                chunk_id=f"g{i}", video_id="other",
                frame_path=str(p), timestamp=float(i),
                embedding=clip.embed_image(str(p)).tolist(),
                caption=f"other frame {i}", tags=[], ocr_text="",
            ))
        res_vq = engine.retrieve("anything", video_id="vq")
        res_other = engine.retrieve("anything", video_id="other")
        assert all(r["video_id"] == "vq" for r in res_vq)
        assert all(r["video_id"] == "other" for r in res_other)


# ===========================================================================
# Smoke: end-to-end pipeline with index + VQA
# ===========================================================================


class TestEndToEnd:

    def test_ingest_then_vqa(self, tmp_path):
        """Full loop: video -> VisionIndexer -> FrameIndex -> VQAEngine."""
        try:
            import cv2  # noqa
        except ImportError:
            pytest.skip("OpenCV not installed")

        video = tmp_path / "e2e.mp4"
        if not _make_dummy_video(video, seconds=4, fps=10):
            pytest.skip("OpenCV video writer unavailable")

        cfg = VLMConfig(
            backend="mock",
            clip_enabled=True,
            ocr_fallback_enabled=False,
            frame_interval_s=1.0,
            max_frames_per_video=10,
            clip_embedding_dim=128,
        )
        vi = VisionIndexer(
            vlm_config=cfg,
            frames_dir=str(tmp_path / "frames"),
            descriptions_dir=str(tmp_path / "desc"),
        )
        chunks = vi.index_video(str(video), video_id="e2e")
        assert len(chunks) >= 1

        idx = InMemoryFrameIndex()
        added = idx.add_chunks(chunks)
        assert added == len(chunks)

        engine = VQAEngine(vlm=vi.vlm, clip=vi.clip, frame_index=idx,
                           top_k_retrieve=5, top_n_answer=2)

        # Visual search
        res = engine.retrieve("red frame", video_id="e2e")
        assert len(res) >= 1

        # VQA
        ans = engine.answer("What colours appear across the video?",
                            video_id="e2e")
        assert ans["answer"]
        assert ans["video_id"] == "e2e"
        assert ans["model"] == "mock" or ans["backend"] == "mock"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
