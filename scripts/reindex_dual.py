# scripts/reindex_dual.py
"""
Script de ré-indexation avec le système dual Nomic + BGE-M3.

Usage :
    python scripts/reindex_dual.py --video data/uploads/video.mp4 --video-id my_video
    python scripts/reindex_dual.py --all        # ré-indexe toutes les vidéos
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def reindex_video(video_id: str, transcription_result, engine, store):
    """Ré-indexe une vidéo avec le double embedding."""
    segments  = [seg.text for seg in transcription_result.segments]
    metadata  = [
        {
            "start_time": seg.start,
            "end_time":   seg.end,
            "speaker":    getattr(seg, "speaker", ""),
            "language":   transcription_result.language,
        }
        for seg in transcription_result.segments
    ]

    if not segments:
        logger.warning(f"No segments for {video_id}")
        return 0

    logger.info(f"Encoding {len(segments)} segments with dual embeddings…")
    dual_embs = engine.encode_passages_batch(segments)

    count = store.add_video_dual(
        video_id=video_id,
        segments=segments,
        dual_embs=dual_embs,
        metadata=metadata,
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex with Dual Embeddings (Nomic + BGE-M3)")
    parser.add_argument("--video",    help="Chemin vers la vidéo")
    parser.add_argument("--video-id", help="ID de la vidéo")
    parser.add_argument("--all",      action="store_true", help="Ré-indexer toutes les vidéos")
    args = parser.parse_args()

    logger.info("=== Initialisation Dual Embedding Engine ===")
    from core.dual_embeddings import DualEmbeddingEngine
    from core.storage import VideoStore

    engine = DualEmbeddingEngine()
    store  = VideoStore()

    logger.info(f"Engine status: {engine.status()}")

    if args.video and args.video_id:
        # ── Mode : une seule vidéo ────────────────────────────────────────────
        logger.info(f"Transcription de {args.video}…")
        from core.transcription import VideoTranscriber
        transcriber = VideoTranscriber()
        result = transcriber.transcribe(args.video)
        count  = reindex_video(args.video_id, result, engine, store)
        logger.info(f"✓ {count} segments indexés pour '{args.video_id}'")

    elif args.all:
        # ── Mode : toutes les vidéos ──────────────────────────────────────────
        videos = store.list_videos()
        logger.info(f"Ré-indexation de {len(videos)} vidéos…")
        for v in videos:
            logger.info(f"  Processing: {v['video_id']}…")
            # À adapter selon votre structure de données

    else:
        # ── Mode : test rapide ────────────────────────────────────────────────
        logger.info("Mode test : encodage de phrases multilingues")
        test_texts = [
            "Neural networks are trained using backpropagation.",          # EN
            "Les réseaux de neurones apprennent par rétropropagation.",    # FR
            "الشبكات العصبية تتعلم باستخدام الانتشار الخلفي.",            # AR
            "Las redes neuronales se entrenan mediante retropropagación.", # ES
            "神经网络通过反向传播进行训练。",                                # ZH
        ]

        logger.info(f"Encoding {len(test_texts)} multilingual passages…")
        results = engine.encode_passages_batch(test_texts)

        for r in results:
            lang = r.language or "?"
            n_sparse = len(r.bge_sparse_indices)
            logger.info(
                f"  [{lang}] nomic={len(r.nomic_dense)}-dim | "
                f"bge_dense={len(r.bge_dense)}-dim | "
                f"bge_sparse={n_sparse} tokens | "
                f"time={r.encode_time_ms}ms"
            )

        logger.info("✓ Test multilingue réussi !")


if __name__ == "__main__":
    main()
