# scripts/migrate_chroma_to_qdrant.py
"""
Migre les données existantes de ChromaDB vers Qdrant.
À exécuter UNE SEULE FOIS après installation de qdrant-client.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage import VideoStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def migrate():
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        logger.warning("⚠️  Aucun répertoire data/processed – migration non nécessaire")
        return

    store = VideoStore()
    migrated = 0

    for emb_file in processed_dir.glob("*_embeddings.json"):
        video_id = emb_file.stem.replace("_embeddings", "")
        logger.info(f"🔄 Migration de '{video_id}'...")

        try:
            with open(emb_file) as f:
                data = json.load(f)

            embeddings = [item["embedding"] for item in data.get("embeddings", [])]
            segments   = [item["segment"]   for item in data.get("embeddings", [])]

            if not embeddings:
                logger.warning(f"  ⚠️  Aucun embedding dans {emb_file.name}")
                continue

            count = store.add_video(video_id, segments, embeddings)
            logger.info(f"  ✅ {count} segments migrés pour '{video_id}'")
            migrated += 1

        except Exception as e:
            logger.error(f"  ❌ Erreur pour '{video_id}': {e}")

    if migrated == 0:
        logger.info("ℹ️  Aucune donnée à migrer (nouvelle installation)")
    else:
        logger.info(f"\n✅ Migration terminée : {migrated} vidéo(s) migrée(s) vers Qdrant")
        logger.info("   Vous pouvez supprimer data/chroma_db/ si tout fonctionne")


if __name__ == "__main__":
    migrate()
