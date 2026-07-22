"""
Diagnostic du problème de recherche
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage import VideoStore
from core.embeddings import EmbeddingGenerator
from typing import Optional


def diagnose():
    print("\n" + "=" * 80)
    print("DIAGNOSTIC")
    print("=" * 80)

    store = VideoStore()

    # 1. Vérifier les vidéos indexées
    print("\n1. Vidéos indexées:")
    videos = store.list_videos()
    for v in videos:
        print(f"   - {v}")

    # 2. Vérifier les métadonnées
    print("\n2. Échantillon de métadonnées:")
    all_data = store.collection.get(limit=3)

    if all_data['metadatas']:
        for i, meta in enumerate(all_data['metadatas'][:3], 1):
            print(f"\n   Segment {i}:")
            print(f"      video_id: {meta.get('video_id')}")
            print(f"      segment_id: {meta.get('segment_id')}")
            print(f"      language: {meta.get('language')}")
            print(f"      timestamp: {meta.get('timestamp_formatted')}")
            print(f"      text: {all_data['documents'][i - 1][:80]}...")

    # 3. Test de recherche simple
    print("\n3. Test de recherche:")

    if videos:
        video_id = videos[0]
        print(f"   Video ID: {video_id}")

        # Générer un embedding de test
        gen = EmbeddingGenerator()
        test_embedding = gen.generate_single("test query about AI and machine learning")

        # Recherche SANS filtre
        print("\n   a) Recherche SANS filtre:")
        results = store.search(test_embedding, video_id=None, language=None, top_k=3)
        print(f"      Résultats: {len(results)}")
        for r in results:
            print(f"         - [{r.timestamp_formatted}] {r.video_id} (score: {r.score:.3f})")

        # Recherche AVEC video_id
        print(f"\n   b) Recherche AVEC video_id={video_id}:")
        results = store.search(test_embedding, video_id=video_id, language=None, top_k=3)
        print(f"      Résultats: {len(results)}")
        for r in results:
            print(f"         - [{r.timestamp_formatted}] {r.text[:60]}... (score: {r.score:.3f})")

        if len(results) == 0:
            print("\n⚠️ PROBLÈME: Aucun résultat avec video_id")
            print("   Cause possible: video_id ne correspond pas")
            print(f"   Essayer de nettoyer et réindexer")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose()
