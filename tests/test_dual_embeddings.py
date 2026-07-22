# tests/test_dual_embeddings.py
"""
Tests pour le système d'embedding dual Nomic + BGE-M3.
"""

import logging
import sys
import os
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_bge_m3():
    print("\n" + "=" * 60)
    print("TEST 1 : BGE-M3 Engine")
    print("=" * 60)

    from core.bge_m3_engine import BGEM3Engine

    engine = BGEM3Engine(use_fp16=True, batch_size=4)
    print(f"✓ Chargé | dim={engine.dim}")

    # Test multilingue
    queries = [
        ("en", "What is machine learning?"),
        ("fr", "Qu'est-ce que l'apprentissage automatique ?"),
        ("ar", "ما هو التعلم الآلي؟"),
        ("de", "Was ist maschinelles Lernen?"),
        ("zh", "什么是机器学习？"),
    ]

    print("\n── Similarité cross-lingue (même question, 5 langues) ──")
    batch_out = engine.encode_batch(
        [q for _, q in queries], is_query=True
    )

    import numpy as np
    en_vec = np.array(batch_out.dense[0])

    for i, (lang, text) in enumerate(queries):
        vec = np.array(batch_out.dense[i])
        sim = float(np.dot(en_vec, vec))
        n_sparse = len(batch_out.sparse[i])
        print(f"  {lang.upper()} | sim_avec_EN={sim:.4f} | sparse_tokens={n_sparse} | '{text[:40]}'")
        assert len(vec) == 1024, f"Dim attendue 1024, obtenu {len(vec)}"
        if i > 0:
            assert sim > 0.70, f"Similarité cross-lingue trop faible: {sim:.4f}"

    print("\n✅ TEST 1 PASSÉ")


def test_dual_engine():
    print("\n" + "=" * 60)
    print("TEST 2 : DualEmbeddingEngine")
    print("=" * 60)

    from core.dual_embeddings import DualEmbeddingEngine

    engine = DualEmbeddingEngine(bge_fallback=True)
    print(f"✓ Status: {engine.status()}")

    # Encoder une requête EN et FR
    for lang, query in [("en", "explain neural networks"), ("fr", "expliquer les réseaux de neurones")]:
        t0 = time.time()
        emb = engine.encode_query(query)
        ms = (time.time() - t0) * 1000
        print(
            f"\n  [{lang}] '{query[:40]}'\n"
            f"    nomic_dense  : {len(emb.nomic_dense)}-dim\n"
            f"    bge_dense    : {len(emb.bge_dense)}-dim\n"
            f"    bge_sparse   : {len(emb.bge_sparse_indices)} tokens\n"
            f"    language det : {emb.language}\n"
            f"    encode time  : {ms:.0f}ms"
        )
        assert len(emb.nomic_dense) == 768
        assert len(emb.bge_dense) == 1024
        assert len(emb.bge_sparse_indices) > 0

    # Batch
    passages = [
        "Machine learning is a subset of artificial intelligence.",
        "L'apprentissage automatique est une branche de l'IA.",
        "الذكاء الاصطناعي يغير العالم.",
    ]
    t0 = time.time()
    batch = engine.encode_passages_batch(passages)
    ms = (time.time() - t0) * 1000
    print(f"\n  Batch de {len(passages)} passages en {ms:.0f}ms")
    assert len(batch) == 3

    print("\n✅ TEST 2 PASSÉ")


def test_qdrant_dual():
    print("\n" + "=" * 60)
    print("TEST 3 : Qdrant collection duale")
    print("=" * 60)

    from core.dual_embeddings import DualEmbeddingEngine, DualEmbedding
    from core.storage import VideoStore

    # Utiliser une collection de test temporaire
    test_collection = "test_dual_temp"

    # Initialiser le store avec la collection de test
    store = VideoStore(collection_name=test_collection)
    engine = DualEmbeddingEngine(bge_fallback=True)

    # Nettoyer si la collection existe déjà
    try:
        store.client.delete_collection(test_collection)
        print("✓ Ancienne collection supprimée")
        time.sleep(1)  # Attendre que la suppression soit effective
    except Exception:
        pass

    # S'assurer que la collection duale est créée
    store._ensure_dual_collection()
    print(f"✓ Collection '{test_collection}' créée")

    # Segments de test multilingues
    test_segments = [
        ("en", "Transformers use self-attention mechanisms to process sequences."),
        ("fr", "Les transformers utilisent l'attention pour traiter les séquences."),
        ("ar", "المحولات تستخدم آلية الانتباه الذاتي لمعالجة التسلسلات."),
        ("es", "Los transformers usan atención propia para procesar secuencias."),
    ]
    texts = [t for _, t in test_segments]
    meta = [{"start_time": i * 10.0, "end_time": i * 10.0 + 9.5, "language": l}
            for i, (l, _) in enumerate(test_segments)]

    print(f"\nEncoding {len(texts)} multilingual segments…")
    dual_embs = engine.encode_passages_batch(texts)

    print(f"Indexing segments...")
    count = store.add_video_dual(
        video_id="test_multilingual",
        segments=texts,
        dual_embs=dual_embs,
        metadata=meta,
        batch_size=2
    )
    print(f"✓ {count} segments indexés")
    assert count == len(texts)

    # Vérifier que les points sont bien indexés
    points_count = store.count()
    print(f"✓ Total points dans la collection: {points_count}")
    assert points_count == len(texts)

    # Recherche en FR → doit trouver FR et EN
    print("\n── Recherche : 'mécanisme d'attention' (FR) ──")
    q_emb = engine.encode_query("mécanisme d'attention dans les transformers")
    results = store.search_dual_hybrid(q_emb, top_k=4)

    for r in results:
        print(f"  [{r['language']}] score={r['score']:.6f} | {r['text'][:60]}")
    assert len(results) > 0

    # Vérifier que les résultats contiennent au moins le segment français et anglais
    languages_found = [r['language'] for r in results]
    assert 'fr' in languages_found or 'FR' in languages_found, "French result not found"
    assert 'en' in languages_found or 'EN' in languages_found, "English result not found"

    # Recherche en AR → doit trouver le segment arabe
    print("\n── Recherche : آلية الانتباه (AR) ──")
    q_emb_ar = engine.encode_query("آلية الانتباه في المحولات")
    results_ar = store.search_dual_hybrid(q_emb_ar, top_k=4)
    for r in results_ar:
        print(f"  [{r['language']}] score={r['score']:.6f} | {r['text'][:60]}")

    # Vérifier que le résultat arabe est présent
    ar_found = any(r['language'] in ['ar', 'AR'] for r in results_ar)
    assert ar_found, "Arabic result not found"

    # Test de filtrage par video_id
    print("\n── Recherche filtrée par video_id ──")
    results_filtered = store.search_dual_hybrid(
        q_emb,
        video_id="test_multilingual",
        top_k=2
    )
    for r in results_filtered:
        print(f"  [{r['language']}] score={r['score']:.6f} | {r['video_id']}")
        assert r['video_id'] == "test_multilingual"

    # Nettoyage
    try:
        store.client.delete_collection(test_collection)
        print("\n✓ Collection de test nettoyée")
    except Exception as e:
        print(f"\n⚠️  Nettoyage impossible: {e}")

    print("\n✅ TEST 3 PASSÉ")


if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("BGE-M3 Engine", test_bge_m3),
        ("Dual Engine", test_dual_engine),
        ("Qdrant Dual", test_qdrant_dual),
    ]:
        try:
            fn()
            results[name] = "✅ PASSÉ"
        except Exception as e:
            results[name] = f"❌ ÉCHOUÉ : {e}"
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    for name, status in results.items():
        print(f"  {status} — {name}")
    print("=" * 60)
