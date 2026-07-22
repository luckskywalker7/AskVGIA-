"""
Réindexation complète de toutes les vidéos
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import UPLOADS_DIR, PROCESSED_DIR
from core.transcription import VideoTranscriber
from core.embeddings import EmbeddingGenerator
from core.storage import VideoStore
import json


def reindex_all():
    print("\n" + "=" * 80)
    print("RÉINDEXATION COMPLÈTE")
    print("=" * 80)

    store = VideoStore()

    # 1. Reset ChromaDB
    print("\n1. Nettoyage ChromaDB...")
    confirm = input("   Confirmer le nettoyage (yes/no)? ")
    if confirm.lower() != 'yes':
        print("   Annulé")
        return

    store.reset()
    print("   ✅ ChromaDB nettoyé")

    # 2. Chercher les transcriptions existantes
    print("\n2. Recherche des transcriptions...")
    transcription_files = list(PROCESSED_DIR.glob("*_transcription.json"))
    print(f"   Trouvé: {len(transcription_files)} fichiers")

    if len(transcription_files) == 0:
        print("   ⚠️ Aucune transcription trouvée")
        print("   Relancez le traitement des vidéos")
        return

    # 3. Réindexer chaque vidéo
    embedding_gen = EmbeddingGenerator()

    for trans_file in transcription_files:
        video_id = trans_file.stem.replace('_transcription', '')
        print(f"\n3. Réindexation: {video_id}")

        # Charger transcription
        with open(trans_file, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)

        segments = trans_data.get('segments', [])
        if not segments:
            print(f"   ⚠️ Aucun segment")
            continue

        print(f"   Segments: {len(segments)}")

        # Générer embeddings
        texts = [seg['text'] for seg in segments]
        batch = embedding_gen.generate_batch(texts, show_progress=False)

        # Indexer
        success = store.add_video(
            video_id=video_id,
            segments=segments,
            embeddings=batch.embeddings,
            overwrite=True
        )

        if success:
            print(f"   ✅ Indexé")
        else:
            print(f"   ❌ Erreur")

    # 4. Vérification
    print("\n4. Vérification finale:")
    stats = store.get_stats()
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Vidéos: {stats['total_videos']}")
    print(f"   IDs: {stats['video_ids']}")

    print("\n" + "=" * 80)
    print("✅ RÉINDEXATION TERMINÉE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    reindex_all()
