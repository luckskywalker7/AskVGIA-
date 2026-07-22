# scripts/run_multilingual_pipeline.py
from __future__ import annotations

import argparse
import inspect
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.transcription import VideoTranscriber
from core.chunking import SemanticChunker
from core.embeddings import EmbeddingGenerator
from core.storage import VideoStore

logger = logging.getLogger("pipeline.multilingual")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class PipelineSummary:
    video_id: str
    video_path: str
    detected_language: str
    language_distribution: dict
    total_segments: int
    total_chunks: int
    total_embedded_chunks: int
    total_indexed: int
    transcription_seconds: float
    chunking_seconds: float
    embedding_seconds: float
    indexing_seconds: float
    total_seconds: float


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def resolve_video_path(video_arg: Optional[str]) -> Path:
    if video_arg:
        p = Path(video_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {p}")
        return p

    candidates = [
        Path("data/uploads/test_video.mp4"),
        Path("data/uploads/videoodiar.mp4"),
        Path("data/uploads/test_video.mov"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    uploads_dir = Path("data/uploads")
    if uploads_dir.exists():
        for file in uploads_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
                return file.resolve()

    raise FileNotFoundError(
        "Aucune vidéo trouvée. Utilise --video data/uploads/nom_fichier.mp4"
    )


def make_video_id(video_path: Path, provided: Optional[str]) -> str:
    if provided and provided.strip():
        return provided.strip()
    return video_path.stem.strip()


def extract_attr(obj: Any, *names: str, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def filter_kwargs_for_callable(method, kwargs: dict) -> Optional[dict]:
    try:
        sig = inspect.signature(method)
    except Exception:
        return {k: v for k, v in kwargs.items() if v is not None}

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if accepts_var_kw:
        return {k: v for k, v in kwargs.items() if v is not None}

    filtered = {}
    required = []

    for name, param in params.items():
        if name == "self":
            continue
        if param.default is inspect._empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            required.append(name)

    for key, value in kwargs.items():
        if value is None:
            continue
        if key in params:
            filtered[key] = value

    for req in required:
        if req not in filtered:
            return None

    return filtered


def try_call_method(method, method_label: str, kw_variants: list[dict], arg_variants: list[tuple]):
    last_type_error = None

    for kwargs in kw_variants:
        filtered = filter_kwargs_for_callable(method, kwargs)
        if filtered is None:
            continue
        try:
            logger.debug("Tentative %s avec kwargs=%s", method_label, list(filtered.keys()))
            return method(**filtered)
        except TypeError as exc:
            last_type_error = exc
            continue

    for args in arg_variants:
        try:
            logger.debug("Tentative %s avec %d args positionnels", method_label, len(args))
            return method(*args)
        except TypeError as exc:
            last_type_error = exc
            continue

    if last_type_error:
        raise last_type_error

    raise RuntimeError(f"Aucune variante d'appel compatible pour {method_label}")


def extract_segments(transcription_result: Any) -> list:
    if transcription_result is None:
        return []

    if isinstance(transcription_result, list):
        return transcription_result

    if isinstance(transcription_result, dict):
        for key in ("segments", "transcript_segments", "items"):
            if key in transcription_result and isinstance(transcription_result[key], list):
                return transcription_result[key]
        return []

    for attr in ("segments", "transcript_segments", "items"):
        value = getattr(transcription_result, attr, None)
        if isinstance(value, list):
            return value

    return []


def extract_chunks(chunking_result: Any) -> list:
    if chunking_result is None:
        return []

    if isinstance(chunking_result, list):
        return chunking_result

    if isinstance(chunking_result, dict):
        for key in ("chunks", "semantic_chunks", "items"):
            if key in chunking_result and isinstance(chunking_result[key], list):
                return chunking_result[key]
        return []

    for attr in ("chunks", "semantic_chunks", "items"):
        value = getattr(chunking_result, attr, None)
        if isinstance(value, list):
            return value

    return []


def extract_language_info(transcription_result: Any) -> tuple[str, dict]:
    language = (
        extract_attr(transcription_result, "language", default=None)
        or extract_attr(transcription_result, "detected_language", default=None)
        or "unknown"
    )

    distribution = (
        extract_attr(transcription_result, "language_distribution", default=None)
        or extract_attr(transcription_result, "distribution", default=None)
        or {}
    )

    if not isinstance(distribution, dict):
        distribution = {}

    return language, distribution


def run_transcription(transcriber: Any, video_path: Path, video_id: str):
    candidate_methods = [
        "process_video",
        "transcribe_video",
        "run_video",
        "process_file",
        "run",
        "process",
        "transcribe",
    ]

    last_exception = None

    for method_name in candidate_methods:
        method = getattr(transcriber, method_name, None)
        if not callable(method):
            continue

        logger.info("Méthode de transcription candidate détectée : %s", method_name)

        if "video" in method_name or method_name in {"process_file", "run", "process"}:
            kw_variants = [
                {"video_path": video_path, "video_id": video_id},
                {"video_path": video_path},
                {"path": video_path, "video_id": video_id},
                {"path": video_path},
                {"file_path": video_path, "video_id": video_id},
                {"file_path": video_path},
                {"input_path": video_path, "video_id": video_id},
                {"input_path": video_path},
                {"video_file": video_path, "video_id": video_id},
                {"video_file": video_path},
                {"video_path": str(video_path), "video_id": video_id},
                {"video_path": str(video_path)},
                {"path": str(video_path), "video_id": video_id},
                {"path": str(video_path)},
            ]
            arg_variants = [
                (video_path, video_id),
                (video_path,),
                (str(video_path), video_id),
                (str(video_path),),
            ]
        else:
            kw_variants = [
                {"audio_path": video_path, "video_id": video_id},
                {"audio_path": video_path},
                {"path": video_path, "video_id": video_id},
                {"path": video_path},
                {"file_path": video_path, "video_id": video_id},
                {"file_path": video_path},
                {"audio_path": str(video_path), "video_id": video_id},
                {"audio_path": str(video_path)},
                {"path": str(video_path), "video_id": video_id},
                {"path": str(video_path)},
            ]
            arg_variants = [
                (video_path, video_id),
                (video_path,),
                (str(video_path), video_id),
                (str(video_path),),
            ]

        try:
            result = try_call_method(
                method=method,
                method_label=f"VideoTranscriber.{method_name}",
                kw_variants=kw_variants,
                arg_variants=arg_variants,
            )
            logger.info("Transcription lancée avec succès via %s", method_name)
            return result

        except TypeError as exc:
            logger.debug("Signature incompatible pour %s : %s", method_name, exc)
            last_exception = exc
            continue
        except Exception as exc:
            logger.exception("Erreur pendant VideoTranscriber.%s : %s", method_name, exc)
            raise

    if last_exception:
        raise RuntimeError(
            f"Impossible d'appeler une méthode de transcription compatible : {last_exception}"
        )

    raise AttributeError(
        "Aucune méthode de transcription trouvée sur VideoTranscriber "
        "(essayé: process_video, transcribe_video, run_video, process_file, run, process, transcribe)"
    )


def run_chunking(
    chunker: Any,
    segments: list,
    transcription_result: Any,
    video_id: str,
    language: str,
):
    """
    IMPORTANT :
    - certaines implémentations attendent l'objet transcription complet
    - d'autres attendent juste la liste des segments
    On privilégie D'ABORD l'objet complet.
    """
    candidate_methods = [
        "chunk",
        "create_semantic_chunks",
        "chunk_segments",
        "create_chunks",
        "process_segments",
        "process",
        "run",
        "generate_chunks",
    ]

    last_exception = None

    for method_name in candidate_methods:
        method = getattr(chunker, method_name, None)
        if not callable(method):
            continue

        logger.info("Méthode de chunking candidate détectée : %s", method_name)

        # Pour chunk/process/run : priorité à l'objet transcription
        if method_name in {"chunk", "process", "run"}:
            kw_variants = [
                {"transcription": transcription_result, "video_id": video_id, "language": language},
                {"transcription_result": transcription_result, "video_id": video_id, "language": language},
                {"transcription": transcription_result, "video_id": video_id},
                {"transcription_result": transcription_result, "video_id": video_id},
                {"transcription": transcription_result},
                {"transcription_result": transcription_result},
                {"segments": segments, "video_id": video_id, "language": language},
                {"segments": segments, "video_id": video_id},
                {"segments": segments},
            ]

            arg_variants = [
                (transcription_result, video_id),
                (transcription_result,),
                (segments, video_id),
                (segments,),
            ]
        else:
            kw_variants = [
                {"segments": segments, "video_id": video_id, "language": language},
                {"segments": segments, "video_id": video_id},
                {"segments": segments, "language": language},
                {"segments": segments},
                {"transcription": transcription_result, "video_id": video_id, "language": language},
                {"transcription_result": transcription_result, "video_id": video_id, "language": language},
                {"transcription": transcription_result},
                {"transcription_result": transcription_result},
            ]

            arg_variants = [
                (segments, video_id),
                (segments,),
                (transcription_result, video_id),
                (transcription_result,),
            ]

        try:
            result = try_call_method(
                method=method,
                method_label=f"SemanticChunker.{method_name}",
                kw_variants=kw_variants,
                arg_variants=arg_variants,
            )
            logger.info("Chunking lancé avec succès via %s", method_name)
            return result

        except TypeError as exc:
            logger.debug("Signature incompatible pour %s : %s", method_name, exc)
            last_exception = exc
            continue
        except Exception as exc:
            logger.exception("Erreur pendant SemanticChunker.%s : %s", method_name, exc)
            raise

    if last_exception:
        raise RuntimeError(
            f"Impossible d'appeler une méthode de chunking compatible : {last_exception}"
        )

    raise AttributeError(
        "Aucune méthode de chunking trouvée sur SemanticChunker "
        "(essayé: chunk, create_semantic_chunks, chunk_segments, create_chunks, process_segments, process, run, generate_chunks)"
    )


def print_summary(summary: PipelineSummary) -> None:
    logger.info("═" * 80)
    logger.info("PIPELINE TERMINÉ")
    logger.info("video_id               : %s", summary.video_id)
    logger.info("video_path             : %s", summary.video_path)
    logger.info("detected_language      : %s", summary.detected_language)
    logger.info("language_distribution  : %s", summary.language_distribution)
    logger.info("segments               : %d", summary.total_segments)
    logger.info("semantic_chunks        : %d", summary.total_chunks)
    logger.info("embedded_chunks        : %d", summary.total_embedded_chunks)
    logger.info("indexed                : %d", summary.total_indexed)
    logger.info("transcription_seconds  : %.2f", summary.transcription_seconds)
    logger.info("chunking_seconds       : %.2f", summary.chunking_seconds)
    logger.info("embedding_seconds      : %.2f", summary.embedding_seconds)
    logger.info("indexing_seconds       : %.2f", summary.indexing_seconds)
    logger.info("total_seconds          : %.2f", summary.total_seconds)
    logger.info("═" * 80)

    print("\n=== DONE ===")
    print(f"video_id: {summary.video_id}")
    print(f"language: {summary.detected_language}")
    print(f"segments: {summary.total_segments}")
    print(f"chunks: {summary.total_chunks}")
    print(f"embedded: {summary.total_embedded_chunks}")
    print(f"indexed: {summary.total_indexed}")
    print(f"total_seconds: {summary.total_seconds:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline multilingue : transcription -> chunking -> embeddings -> Qdrant"
    )
    parser.add_argument("--video", type=str, default=None, help="Chemin vers la vidéo à traiter")
    parser.add_argument("--video-id", type=str, default=None, help="Identifiant logique de la vidéo")
    parser.add_argument("--collection", type=str, default=None, help="Nom de collection Qdrant")
    parser.add_argument("--reset-store", action="store_true", help="Réinitialise la collection avant indexation")
    parser.add_argument("--verbose", action="store_true", help="Logs détaillés")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    total_start = time.perf_counter()

    try:
        video_path = resolve_video_path(args.video)
        video_id = make_video_id(video_path, args.video_id)

        logger.info("🚀 Lancement pipeline multilingue")
        logger.info("video_path = %s", video_path)
        logger.info("video_id   = %s", video_id)

        transcriber = VideoTranscriber()
        chunker = SemanticChunker()
        embedder = EmbeddingGenerator()

        store = VideoStore(collection_name=args.collection) if args.collection else VideoStore()

        if args.reset_store:
            logger.warning("Réinitialisation de la collection demandée")
            store.reset()

        # 1) Transcription
        logger.info("Étape 1/4 — Transcription")
        t0 = time.perf_counter()
        transcription_result = run_transcription(
            transcriber=transcriber,
            video_path=video_path,
            video_id=video_id,
        )
        transcription_seconds = time.perf_counter() - t0

        segments = extract_segments(transcription_result)
        if not segments:
            raise RuntimeError("Aucun segment produit par la transcription")

        detected_language, language_distribution = extract_language_info(transcription_result)

        logger.info("Transcription OK : %d segments", len(segments))
        logger.info("Langue détectée : %s | distribution=%s", detected_language, language_distribution)

        # 2) Chunking
        logger.info("Étape 2/4 — Semantic chunking")
        t1 = time.perf_counter()
        chunking_result = run_chunking(
            chunker=chunker,
            segments=segments,
            transcription_result=transcription_result,
            video_id=video_id,
            language=detected_language,
        )
        chunking_seconds = time.perf_counter() - t1

        semantic_chunks = extract_chunks(chunking_result)
        if not semantic_chunks:
            raise RuntimeError("Aucun chunk sémantique produit")

        logger.info("Chunking OK : %d chunks", len(semantic_chunks))

        # 3) Embeddings
        logger.info("Étape 3/4 — Génération des embeddings")
        t2 = time.perf_counter()
        embedded_chunks = embedder.generate_chunks(semantic_chunks, video_id)
        embedding_seconds = time.perf_counter() - t2

        if not embedded_chunks:
            raise RuntimeError("Aucun EmbeddedChunk généré")

        logger.info("Embeddings OK : %d chunks embarqués", len(embedded_chunks))

        # 4) Indexation
        logger.info("Étape 4/4 — Indexation Qdrant")
        t3 = time.perf_counter()
        indexed = store.add_embedded_chunks(embedded_chunks)
        indexing_seconds = time.perf_counter() - t3

        total_seconds = time.perf_counter() - total_start

        summary = PipelineSummary(
            video_id=video_id,
            video_path=str(video_path),
            detected_language=detected_language,
            language_distribution=language_distribution,
            total_segments=len(segments),
            total_chunks=len(semantic_chunks),
            total_embedded_chunks=len(embedded_chunks),
            total_indexed=indexed,
            transcription_seconds=transcription_seconds,
            chunking_seconds=chunking_seconds,
            embedding_seconds=embedding_seconds,
            indexing_seconds=indexing_seconds,
            total_seconds=total_seconds,
        )

        print_summary(summary)
        return 0

    except Exception as exc:
        total_seconds = time.perf_counter() - total_start
        logger.exception("❌ Pipeline échoué après %.2fs : %s", total_seconds, exc)
        print("\n=== FAILED ===")
        print(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
