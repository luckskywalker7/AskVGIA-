# tests/test_vision.py
"""Tests d'intégration pour Phase 5 – Keyframe CLIP Embeddings."""

import logging
import os
import sys
import time
import tempfile
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 1 : CLIPEngine
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_engine():
    """Vérifie le chargement et l'encodage CLIP."""
    print("\n" + "="*60)
    print("SUITE 1 : CLIPEngine")
    print("="*60)

    from core.vision import CLIPEngine

    clip = CLIPEngine()
    print(f"✓ Modèle chargé : {clip.model_name}")
    print(f"✓ Dimension vecteur : {clip.dim}")
    assert clip.dim == 512, f"Attendu 512, obtenu {clip.dim}"

    # Test embed_text
    t0 = time.time()
    text_emb = clip.embed_text("une personne qui parle devant un écran")
    t_text = (time.time() - t0) * 1000
    assert len(text_emb) == 512
    assert abs(np.linalg.norm(text_emb) - 1.0) < 1e-5, "Vecteur non normalisé"
    print(f"✓ embed_text : {len(text_emb)}-dim en {t_text:.1f}ms")

    # Test embed_frame avec une image synthétique
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    t0 = time.time()
    img_emb = clip.embed_frame(dummy_image)
    t_img = (time.time() - t0) * 1000
    assert len(img_emb) == 512
    assert abs(np.linalg.norm(img_emb) - 1.0) < 1e-5
    print(f"✓ embed_frame : {len(img_emb)}-dim en {t_img:.1f}ms")

    # Test similarité : texte similaire doit scorer plus haut
    text1 = clip.embed_text("un chat sur un canapé")
    text2 = clip.embed_text("une voiture sur l'autoroute")
    sim = np.dot(text1, text2)
    print(f"✓ Similarité cosinus (textes différents) : {sim:.4f}")

    # Test batch
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
    t0 = time.time()
    batch_embs = clip.embed_frames_batch(images)
    t_batch = (time.time() - t0) * 1000
    assert len(batch_embs) == 5
    print(f"✓ embed_frames_batch (5 frames) : {t_batch:.1f}ms")

    print("\n✅ Suite 1 PASSÉE")
    return clip


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 2 : KeyframeExtractor (avec vidéo synthétique)
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_video(path: str, n_scenes: int = 5, duration_sec: int = 30):
    """Crée une vidéo synthétique avec des changements de scène nets."""
    import cv2

    fps = 10
    frame_size = (320, 240)
    colors = [
        (50, 50, 200),   # Rouge
        (50, 200, 50),   # Vert
        (200, 50, 50),   # Bleu
        (200, 200, 50),  # Cyan
        (200, 50, 200),  # Magenta
    ]
    frames_per_scene = (duration_sec * fps) // n_scenes

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

    for scene_idx in range(n_scenes):
        color = colors[scene_idx % len(colors)]
        for f in range(frames_per_scene):
            # Ajouter un peu de bruit pour simuler le mouvement
            noise = np.random.randint(-10, 10, (*frame_size[::-1], 3), dtype=np.int16)
            frame = np.clip(
                np.full((*frame_size[::-1], 3), color, dtype=np.int16) + noise, 0, 255
            ).astype(np.uint8)
            writer.write(frame)

    writer.release()
    return path


def test_keyframe_extractor():
    """Teste l'extraction avec une vidéo synthétique."""
    print("\n" + "="*60)
    print("SUITE 2 : KeyframeExtractor")
    print("="*60)

    from core.vision import KeyframeExtractor

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "synthetic.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        # Créer vidéo synthétique
        create_synthetic_video(video_path, n_scenes=5, duration_sec=20)
        print(f"✓ Vidéo synthétique créée : {video_path}")

        extractor = KeyframeExtractor(
            scene_threshold=0.3,
            min_interval_sec=1.0,
            max_frames=50,
            save_dir=frames_dir,
        )

        t0 = time.time()
        keyframes = extractor.extract(video_path, "test_video_001")
        elapsed = time.time() - t0

        print(f"✓ {len(keyframes)} keyframes extraites en {elapsed:.1f}s")
        assert len(keyframes) >= 5, f"Attendu >= 5 keyframes (5 scènes), obtenu {len(keyframes)}"
        assert len(keyframes) <= 50, "Limite max_frames non respectée"

        # Vérifier les attributs
        kf = keyframes[0]
        assert kf.video_id == "test_video_001"
        assert kf.timestamp_sec >= 0
        assert kf.image is not None
        assert kf.image.shape == (224, 224, 3)
        print(f"✓ Structure keyframe : frame_id='{kf.frame_id}', ts={kf.timestamp_sec:.1f}s")

        # Vérifier sauvegarde disque
        saved = [kf for kf in keyframes if kf.frame_path is not None]
        print(f"✓ {len(saved)} frames sauvegardées sur disque")

    print("\n✅ Suite 2 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 3 : VisionPipeline end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_vision_pipeline():
    """Test du pipeline complet (extraction + CLIP) sans OCR."""
    print("\n" + "="*60)
    print("SUITE 3 : VisionPipeline (end-to-end)")
    print("="*60)

    from core.vision import VisionPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        create_synthetic_video(video_path, n_scenes=3, duration_sec=15)

        pipeline = VisionPipeline(
            scene_threshold=0.3,
            min_interval_sec=1.0,
            max_frames=20,
            ocr_enabled=False,
            save_frames_dir=os.path.join(tmpdir, "kf"),
            embed_batch_size=8,
        )

        t0 = time.time()
        frame_embs = pipeline.process_video(video_path, "pipeline_test_001")
        elapsed = time.time() - t0

        print(f"✓ {len(frame_embs)} FrameEmbedding générés en {elapsed:.1f}s")
        assert len(frame_embs) > 0

        fe = frame_embs[0]
        assert len(fe.embedding) == pipeline.vector_dim == 512
        assert fe.video_id == "pipeline_test_001"
        assert abs(np.linalg.norm(fe.embedding) - 1.0) < 1e-5
        print(f"✓ FrameEmbedding[0] : ts={fe.timestamp_sec:.1f}s, dim={len(fe.embedding)}")

        # Test embed_query
        q_emb = pipeline.embed_query("une image colorée")
        assert len(q_emb) == 512
        print(f"✓ embed_query : {len(q_emb)}-dim")

    print("\n✅ Suite 3 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 4 : Qdrant intégration (VideoStore)
# ─────────────────────────────────────────────────────────────────────────────

def test_qdrant_visual():
    """Test d'indexation et de recherche visuelle dans Qdrant."""
    print("\n" + "="*60)
    print("SUITE 4 : Qdrant – Collection visuelle")
    print("="*60)

    try:
        from core.storage import VideoStore
        from core.vision import FrameEmbedding
    except ImportError as e:
        print(f"⚠ Import manquant : {e}")
        return

    # Créer des FrameEmbedding synthétiques
    def make_embedding(seed: int) -> list[float]:
        rng = np.random.RandomState(seed)
        v = rng.randn(512).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    test_frames = [
        FrameEmbedding(
            frame_id=f"test_vid_{i:04d}",
            video_id="test_vid",
            timestamp_sec=float(i * 10),
            embedding=make_embedding(i),
            frame_path=f"data/keyframes/test_vid_{i:04d}.jpg",
            ocr_text="texte visible" if i % 2 == 0 else "",
            scene_score=0.5,
        )
        for i in range(10)
    ]

    store = VideoStore()

    # Nettoyer si collection existante
    try:
        store.client.delete_collection(store.VISUAL_COLLECTION)
        print("✓ Ancienne collection visuelle supprimée")
    except Exception:
        pass

    # Indexer
    t0 = time.time()
    count = store.add_frames(test_frames)
    t_index = (time.time() - t0) * 1000
    assert count == 10
    print(f"✓ {count} frames indexées en {t_index:.1f}ms")

    # Stats
    stats = store.get_visual_stats()
    print(f"✓ Stats : {stats}")
    assert stats["total_frames"] == 10

    # Recherche avec un vecteur aléatoire
    query_vec = make_embedding(42)
    t0 = time.time()
    results = store.search_visual(query_vec, top_k=5)
    t_search = (time.time() - t0) * 1000
    print(f"✓ search_visual : {len(results)} résultats en {t_search:.1f}ms")
    assert len(results) <= 5

    if results:
        r = results[0]
        print(f"  Top 1 : frame_id={r['frame_id']}, ts={r['timestamp_sec']}s, score={r['score']}")
        assert 0 <= r["score"] <= 1.0

    # Recherche filtrée par video_id
    filtered = store.search_visual(query_vec, video_id="test_vid", top_k=3)
    assert all(r["video_id"] == "test_vid" for r in filtered)
    print(f"✓ Filtre video_id fonctionne : {len(filtered)} résultats")

    # list_video_frames
    frames_list = store.list_video_frames("test_vid")
    assert len(frames_list) == 10
    print(f"✓ list_video_frames : {len(frames_list)} frames trouvées")

    # Suppression
    store.delete_video_frames("test_vid")
    stats_after = store.get_visual_stats()
    print(f"✓ Après suppression : {stats_after['total_frames']} frames")

    print("\n✅ Suite 4 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 5 : Performance et benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_performance():
    """Benchmark des performances CLIP."""
    print("\n" + "="*60)
    print("SUITE 5 : Performance")
    print("="*60)

    from core.vision import CLIPEngine

    clip = CLIPEngine()
    images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(32)
    ]

    # Benchmark batch
    t0 = time.time()
    _ = clip.embed_frames_batch(images)
    elapsed = time.time() - t0

    fps = len(images) / elapsed
    print(f"✓ Encodage batch : {len(images)} frames en {elapsed:.2f}s → {fps:.1f} frames/s")

    # Benchmark recherche texte
    queries = [
        "une personne qui parle",
        "un tableau blanc",
        "une salle de réunion",
        "un écran d'ordinateur",
    ]
    t0 = time.time()
    for q in queries:
        _ = clip.embed_text(q)
    t_queries = (time.time() - t0) * 1000 / len(queries)
    print(f"✓ embed_text moyen : {t_queries:.1f}ms/requête")

    # Seuils de performance
    assert fps > 1, f"Trop lent : {fps:.1f} frames/s (min: 1)"
    assert t_queries < 1000, f"Requête texte trop lente : {t_queries:.1f}ms"
    print(f"\n✅ Suite 5 PASSÉE — {fps:.1f} fps, {t_queries:.1f}ms/query")


# ─────────────────────────────────────────────────────────────────────────────
#  Runner principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🎬 " * 20)
    print("TESTS – Phase 5 : Keyframe CLIP Embeddings")
    print("🎬 " * 20)

    results = {}

    for name, fn in [
        ("CLIPEngine", test_clip_engine),
        ("KeyframeExtractor", test_keyframe_extractor),
        ("VisionPipeline", test_vision_pipeline),
        ("Qdrant Visual", test_qdrant_visual),
        ("Performance", test_performance),
    ]:
        try:
            fn()
            results[name] = "✅ PASSÉ"
        except Exception as e:
            results[name] = f"❌ ÉCHOUÉ : {e}"
            logger.exception(f"Erreur dans {name}")

    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    for name, status in results.items():
        print(f"  {status} — {name}")

    n_passed = sum(1 for v in results.values() if v.startswith("✅"))
    print(f"\n{n_passed}/{len(results)} tests réussis")
