"""
core/transcription.py — WhisperX-based video transcriber v3.1
==============================================================
Corrections v3.0 → v3.1 :

 BUG-4  DiarizationPipeline.__init__() got unexpected keyword argument
        'use_auth_token'. WhisperX ≥3.7 / pyannote ≥4.0 a remplacé
        use_auth_token par token.
        FIX : essaie d'abord token=, puis fallback use_auth_token= pour
        compatibilité avec les anciennes versions. Log explicite.

 (v3.0 fixes preserved) :
 BUG-1  _confidence_filter() hallucination guard (conf >= 0.3 pour >= 8 mots)
 BUG-2  Arabic-aware Jaccard deduplication
 BUG-3  micro-chunk guard (< 5s → skip)

 BONUS  Safe settings/imports. Aliases de compatibilité.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# FFmpeg path resolution
# On Windows, ffmpeg must be the full-shared build and in PATH.
# Common install locations are tried as fallbacks if PATH lookup fails.
# ──────────────────────────────────────────────────────────────────────

import shutil as _shutil
import sys as _sys

def _find_ffmpeg() -> str:
    """Return the ffmpeg executable path, searching PATH then common locations."""
    found = _shutil.which("ffmpeg")
    if found:
        return found
    # Common Windows install locations
    candidates = [
        r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\bin",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "ffmpeg"  # let subprocess raise the original FileNotFoundError

_FFMPEG_EXE = _find_ffmpeg()

# ──────────────────────────────────────────────────────────────────────
# Safe settings
# ──────────────────────────────────────────────────────────────────────

try:
    from config.settings import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any = None) -> Any:
    if _settings is None:
        return default
    return getattr(_settings, name, default)


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(_cfg(name, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(name: str, default: bool) -> bool:
    val = _cfg(name, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


# ──────────────────────────────────────────────────────────────────────
# Safe imports : contracts
# ──────────────────────────────────────────────────────────────────────

try:
    from core.contracts import TranscriptSegment, TranscriptionResult
except ImportError:
    logger.info("core.contracts not found — using local dataclasses")

    @dataclass
    class TranscriptSegment:
        text: str = ""
        start_time: float = 0.0
        end_time: float = 0.0
        language: str = "unknown"
        speaker: str = "SPEAKER_00"
        confidence: float = 1.0
        words: list = field(default_factory=list)

    @dataclass
    class TranscriptionResult:
        video_id: str = ""
        segments: List[Any] = field(default_factory=list)
        language: str = "unknown"
        language_distribution: Dict[str, float] = field(default_factory=dict)
        total_segments: int = 0
        duration_seconds: float = 0.0
        has_diarization: bool = False
        processing_seconds: float = 0.0
        audio_path: str = ""


# ──────────────────────────────────────────────────────────────────────
# BUG-2 FIX : Arabic-aware tokenizer for deduplication
# ──────────────────────────────────────────────────────────────────────

_external_tokenize = None
try:
    from core.storage import tokenize as _external_tokenize
    logger.debug("Using tokenize() from core.storage")
except ImportError:
    pass

_AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_AR_TATWEEL = re.compile(r"\u0640+")


def _tokenize_for_similarity(text: str, language: str = "en") -> List[str]:
    if _external_tokenize is not None:
        return _external_tokenize(text, language)
    if not text:
        return []
    text = text.strip().lower()
    if language in ("ar", "fa", "ur"):
        text = _AR_DIACRITICS.sub("", text)
        text = _AR_TATWEEL.sub("", text)
        text = text.replace("\u0623", "\u0627")
        text = text.replace("\u0625", "\u0627")
        text = text.replace("\u0622", "\u0627")
        text = text.replace("\u0629", "\u0647")
        tokens = re.split(r"[\s\u200c\u200d.,;:!?\-/\\()«»\"']+", text)
    else:
        tokens = re.split(r"[\s.,;:!?\-/\\()«»\"']+", text)
    return [t for t in tokens if len(t) >= 2]


# ──────────────────────────────────────────────────────────────────────
# Language detector (lazy, multi-backend)
# ──────────────────────────────────────────────────────────────────────

_lang_detector_cache: Optional[Any] = None
_lang_detector_loaded = False


def _get_language_detector() -> Optional[Any]:
    global _lang_detector_cache, _lang_detector_loaded
    if _lang_detector_loaded:
        return _lang_detector_cache
    _lang_detector_loaded = True

    try:
        from core.language_detector import LanguageDetector
        _lang_detector_cache = ("core", LanguageDetector.get())
        logger.info("Language detector: core.language_detector")
        return _lang_detector_cache
    except Exception:
        pass

    try:
        from lingua import Language, LanguageDetectorBuilder
        det = (
            LanguageDetectorBuilder
            .from_languages(Language.ARABIC, Language.ENGLISH, Language.FRENCH)
            .with_minimum_relative_distance(0.25)
            .build()
        )
        _lang_detector_cache = ("lingua", det)
        logger.info("Language detector: lingua (ar/en/fr)")
        return _lang_detector_cache
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("lingua init failed: %s", exc)

    try:
        import langdetect
        _lang_detector_cache = ("langdetect", langdetect)
        logger.info("Language detector: langdetect")
        return _lang_detector_cache
    except ImportError:
        pass

    logger.info("No language detector available")
    return None


def _detect_segment_language(text: str, detector: Any) -> Optional[str]:
    try:
        if not text or len(text.strip()) < 10:
            return None
        backend, det = detector
        if backend == "core":
            result = det.detect_language(text)
            if isinstance(result, str):
                return result.lower()
            code = getattr(result, "code", None) or getattr(result, "language", None)
            return str(code).lower() if code else None
        elif backend == "lingua":
            from lingua import Language
            result = det.detect_language_of(text)
            if result is not None:
                _map = {Language.ARABIC: "ar", Language.ENGLISH: "en", Language.FRENCH: "fr"}
                return _map.get(result)
        elif backend == "langdetect":
            code = det.detect(text)
            return str(code).lower() if code else None
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

CHUNK_DURATION_S = 300
CHUNK_OVERLAP_S = 10
MIN_CHUNK_DURATION_S = 5.0
MIN_CONFIDENCE = 0.55
MIN_CONFIDENCE_LONG = 0.30
MIN_WORDS = 2
MIN_WORDS_LONG = 8


# ══════════════════════════════════════════════════════════════════════
# VideoTranscriber
# ══════════════════════════════════════════════════════════════════════

class VideoTranscriber:
    """
    WhisperX-based video transcriber v3.1.
    BUG-4 FIX: DiarizationPipeline uses token= (not use_auth_token=).
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        hf_token: Optional[str] = None,
        diarization_enabled: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_dir: Optional[str] = None,
        load_on_init: bool = False,
        **_: Any,
    ):
        self.model_size = model_size or _cfg("whisper_model", "large-v2")
        self.device = device or _cfg("whisper_device", "cpu")
        self.compute_type = compute_type or _cfg("whisper_compute_type", "int8")
        self.hf_token = hf_token or _cfg("effective_hf_token", None)
        self.diarization_enabled = (
            diarization_enabled if diarization_enabled is not None
            else _cfg_bool("pyannote_enabled", False)
        )
        self.min_speakers = (
            min_speakers if min_speakers is not None
            else _cfg_int("pyannote_min_speakers", 1)
        )
        self.max_speakers = (
            max_speakers if max_speakers is not None
            else _cfg_int("pyannote_max_speakers", 5)
        )

        self.audio_dir = Path(audio_dir or _cfg("audio_dir", "data/audio"))
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.whisper_model = None
        self._diarization_pipeline = None

        logger.info(
            "VideoTranscriber v3.1 | model=%s | device=%s | diarization=%s",
            self.model_size, self.device,
            "on" if self.diarization_enabled else "off",
        )

        if load_on_init:
            self._load_whisper()

    # ==================================================================
    # Public API
    # ==================================================================

    def process_video(
        self,
        video_path: Any,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        **_kwargs: Any,
    ) -> TranscriptionResult:
        video_path = Path(str(video_path))
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not video_id:
            video_id = video_path.stem

        t0 = time.time()
        logger.info("▶ Transcription START video_id=%s", video_id)

        audio_path = self._extract_audio(video_path=video_path, video_id=video_id)
        audio_duration, audio_sample_rate = self._validate_audio(audio_path)
        logger.info("  Audio: duration=%.1fs sr=%dHz", audio_duration, audio_sample_rate)

        whisper_result = self._transcribe_smart(audio_path, audio_duration)
        aligned_result = self._align(whisper_result=whisper_result, audio_path=audio_path)
        aligned_result, has_diarization = self._run_diarization(
            audio_path=audio_path, aligned_result=aligned_result,
        )

        file_language = language or whisper_result.get("language") or "unknown"
        segments = self._merge_transcript(whisper_result=aligned_result, file_language=file_language)
        segments = self._detect_and_fix_languages(segments, file_language)

        before_filter = len(segments)
        segments = self._confidence_filter(segments)
        if before_filter != len(segments):
            logger.info("  Filter: %d → %d (dropped %d)", before_filter, len(segments), before_filter - len(segments))

        dominant_language, language_distribution = self._compute_language_distribution(
            fallback_language=file_language, segments=segments,
        )

        duration_seconds = self._compute_duration(segments)
        elapsed = time.time() - t0

        logger.info(
            "✅ Transcription DONE | video_id=%s | segments=%d | lang=%s | %.1fs",
            video_id, len(segments), dominant_language, elapsed,
        )

        return TranscriptionResult(
            video_id=video_id,
            segments=segments,
            language=dominant_language,
            language_distribution=language_distribution,
            total_segments=len(segments),
            duration_seconds=duration_seconds,
            has_diarization=has_diarization,
            processing_seconds=elapsed,
            audio_path=str(audio_path),
        )

    # Aliases
    def transcribe(self, *a, **kw) -> TranscriptionResult:
        return self.process_video(*a, **kw)

    def transcribe_video(self, *a, **kw) -> TranscriptionResult:
        return self.process_video(*a, **kw)

    def run(self, *a, **kw) -> TranscriptionResult:
        return self.process_video(*a, **kw)

    def process(self, *a, **kw) -> TranscriptionResult:
        return self.process_video(*a, **kw)

    # ==================================================================
    # 1. Audio extraction
    # ==================================================================

    def _extract_audio(self, video_path: Path, video_id: str) -> Path:
        audio_path = self.audio_dir / f"{video_id}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 1000:
            logger.info("  Audio already extracted: %s", audio_path.name)
            return audio_path

        cmd = [
            _FFMPEG_EXE, "-y", "-i", str(video_path),
            "-ar", "16000", "-ac", "1", "-vn", "-f", "wav",
            str(audio_path),
        ]
        logger.info("  Extracting audio: %s", video_path.name)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg not installed or not in PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg timed out (>15min).") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {(proc.stderr or '')[-500:]}")
        if not audio_path.exists() or audio_path.stat().st_size < 1000:
            raise RuntimeError(f"Invalid extracted audio: {audio_path}")
        return audio_path

    # ==================================================================
    # 2. Audio validation
    # ==================================================================

    def _validate_audio(self, audio_path: Path) -> Tuple[float, int]:
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            if info.duration < 0.5:
                raise RuntimeError(f"Audio too short: {info.duration:.2f}s")
            if info.samplerate < 8000:
                raise RuntimeError(f"Sample rate too low: {info.samplerate}Hz")
            if info.duration > 14400:
                logger.warning("Very long audio: %.0fs (%.1fh)", info.duration, info.duration / 3600)
            return info.duration, info.samplerate
        except ImportError:
            logger.warning("soundfile not installed, skipping validation")
            return 0.0, 16000
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Audio validation failed: %s", exc)
            return 0.0, 16000

    # ==================================================================
    # 3. Transcription
    # ==================================================================

    def _load_whisper(self) -> None:
        if self.whisper_model is not None:
            return
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError("WhisperX not installed.") from exc
        logger.info("  Loading WhisperX | size=%s | device=%s", self.model_size, self.device)
        # Anti-hallucination ASR options. On music/silence/ambient audio Whisper
        # invents speech (repeated phrases, "thank you", subtitle credits). These
        # thresholds make it emit nothing instead of hallucinating:
        #   no_speech_threshold  — if no-speech prob > this, treat segment as silence
        #   log_prob_threshold   — drop low-logprob (uncertain) decodes
        #   compression_ratio_threshold — drop repetitive/looping text
        #   condition_on_previous_text=False — stop hallucination loops feeding themselves
        asr_options = {
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": False,
            "temperatures": [0.0],
        }
        try:
            self.whisper_model = whisperx.load_model(
                self.model_size, self.device, compute_type=self.compute_type,
                asr_options=asr_options,
                vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
            )
        except TypeError:
            # older/newer whisperx signature without these kwargs — fall back
            logger.warning("  WhisperX load_model: asr_options unsupported, using defaults")
            self.whisper_model = whisperx.load_model(
                self.model_size, self.device, compute_type=self.compute_type,
            )

    def _ensure_whisper_loaded(self) -> None:
        if self.whisper_model is None:
            self._load_whisper()

    def release_gpu(self) -> None:
        """
        Unload WhisperX + alignment + diarization models from VRAM.

        Call this immediately after transcription, before VLM starts.
        Frees ~3.0 GB so qwen3-vl:8b gets full VRAM and runs at
        ~3-4 s/frame instead of ~25 s/frame (CPU spill).
        Safe to call multiple times (idempotent).
        """
        try:
            import torch
            import gc as _gc
        except ImportError:
            return

        released = []

        if getattr(self, "whisper_model", None) is not None:
            try:
                del self.whisper_model
                self.whisper_model = None
                released.append("whisper_model")
            except Exception as e:
                logger.warning("release_gpu: whisper_model: %s", e)

        for attr in ("align_model", "_align_model", "alignment_model"):
            if getattr(self, attr, None) is not None:
                try:
                    delattr(self, attr)
                    released.append(attr)
                except Exception:
                    pass

        for attr in ("diarize_model", "_diarize_model", "diarization_pipeline"):
            if getattr(self, attr, None) is not None:
                try:
                    delattr(self, attr)
                    released.append(attr)
                except Exception:
                    pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            logger.info(
                "release_gpu: released %s — %.1f GB VRAM now free",
                released, free_gb,
            )
        else:
            logger.info("release_gpu: released %s (CPU mode)", released)

        _gc.collect()

    def _transcribe_smart(self, audio_path: Path, audio_duration: float) -> Dict[str, Any]:
        if 0 < audio_duration <= CHUNK_DURATION_S:
            return self._transcribe(audio_path)
        elif audio_duration > CHUNK_DURATION_S:
            return self._transcribe_in_chunks(audio_path, audio_duration)
        else:
            return self._transcribe(audio_path)

    def _transcribe(self, audio_path: Path) -> Dict[str, Any]:
        self._ensure_whisper_loaded()
        import whisperx
        logger.info("  WhisperX transcription: %s", audio_path.name)
        audio = whisperx.load_audio(str(audio_path))
        result = self.whisper_model.transcribe(audio, batch_size=16)
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected WhisperX output.")
        result.setdefault("segments", [])
        result.setdefault("language", "unknown")
        return result

    def _transcribe_in_chunks(self, audio_path: Path, total_duration: float) -> Dict[str, Any]:
        self._ensure_whisper_loaded()
        import whisperx
        logger.info("  Chunked transcription | total=%.1fs", total_duration)

        full_audio = whisperx.load_audio(str(audio_path))
        sample_rate = 16000
        all_segments: List[Dict[str, Any]] = []
        detected_language = "unknown"
        offset = 0.0
        chunk_idx = 0

        while offset < total_duration:
            chunk_end = min(offset + CHUNK_DURATION_S, total_duration)
            chunk_len = chunk_end - offset

            if chunk_len < MIN_CHUNK_DURATION_S:
                logger.debug("  Skipping micro-chunk at %.1fs (%.1fs)", offset, chunk_len)
                break

            start_sample = int(offset * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            chunk_audio = full_audio[start_sample:end_sample]

            if len(chunk_audio) < sample_rate:
                break

            logger.info("  Chunk %d: %.1fs → %.1fs", chunk_idx, offset, chunk_end)

            try:
                chunk_result = self.whisper_model.transcribe(chunk_audio, batch_size=16)
            except Exception as exc:
                logger.warning("  Chunk %d failed: %s", chunk_idx, exc)
                offset += CHUNK_DURATION_S - CHUNK_OVERLAP_S
                chunk_idx += 1
                continue

            if chunk_idx == 0:
                detected_language = chunk_result.get("language", "unknown")

            for seg in chunk_result.get("segments", []):
                seg["start"] = float(seg.get("start", 0)) + offset
                seg["end"] = float(seg.get("end", 0)) + offset
                for w in seg.get("words", []):
                    if "start" in w and w["start"] is not None:
                        w["start"] = float(w["start"]) + offset
                    if "end" in w and w["end"] is not None:
                        w["end"] = float(w["end"]) + offset
                all_segments.append(seg)

            offset += CHUNK_DURATION_S - CHUNK_OVERLAP_S
            chunk_idx += 1

        all_segments = self._deduplicate_overlap_segments(all_segments, detected_language)
        logger.info("  Chunked done | chunks=%d | segments=%d", chunk_idx, len(all_segments))
        return {"segments": all_segments, "language": detected_language}

    # ==================================================================
    # Deduplication (BUG-2 FIX)
    # ==================================================================

    def _deduplicate_overlap_segments(
        self, segments: List[Dict[str, Any]], language: str = "en",
    ) -> List[Dict[str, Any]]:
        if len(segments) <= 1:
            return segments
        segments.sort(key=lambda s: float(s.get("start", 0)))
        deduped: List[Dict[str, Any]] = [segments[0]]
        for seg in segments[1:]:
            prev = deduped[-1]
            time_diff = abs(float(seg.get("start", 0)) - float(prev.get("start", 0)))
            if time_diff < 2.0:
                text_a = str(prev.get("text", "")).strip()
                text_b = str(seg.get("text", "")).strip()
                seg_lang = str(seg.get("language") or prev.get("language") or language)
                if self._tokenized_similarity(text_a, text_b, seg_lang) > 0.8:
                    if len(text_b.split()) > len(text_a.split()):
                        deduped[-1] = seg
                    continue
            deduped.append(seg)
        return deduped

    @staticmethod
    def _tokenized_similarity(a: str, b: str, language: str = "en") -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        tokens_a = set(_tokenize_for_similarity(a, language))
        tokens_b = set(_tokenize_for_similarity(b, language))
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    # ==================================================================
    # 4. Alignment
    # ==================================================================

    def _align(self, whisper_result: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
        try:
            import whisperx
        except ImportError:
            logger.warning("WhisperX alignment unavailable")
            return whisper_result
        language_code = whisper_result.get("language", "en")
        try:
            logger.info("  WhisperX alignment | language=%s", language_code)
            audio = whisperx.load_audio(str(audio_path))
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            aligned = whisperx.align(
                whisper_result["segments"], model_a, metadata,
                audio, self.device, return_char_alignments=False,
            )
            aligned["language"] = language_code
            return aligned
        except Exception as exc:
            logger.warning("  Alignment failed: %s", exc)
            return whisper_result

    # ==================================================================
    # 5. BUG-4 FIX : Diarization — token= instead of use_auth_token=
    # ==================================================================

    def _run_diarization(
        self, audio_path: Path, aligned_result: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """
        BUG-4 FIX :
        - WhisperX >= 3.7 / pyannote >= 4.0 uses `token=` (not `use_auth_token=`)
        - Try `token=` first, then fallback to `use_auth_token=` for older versions
        - Log which parameter worked
        """
        if not self.diarization_enabled:
            return aligned_result, False

        if not self.hf_token:
            logger.warning("  HF token missing, diarization disabled.")
            return aligned_result, False

        try:
            import whisperx
        except ImportError:
            logger.warning("  WhisperX not available for diarization.")
            return aligned_result, False

        # ── BUG-4 FIX : load pipeline with correct parameter ─────────
        try:
            if self._diarization_pipeline is None:
                self._diarization_pipeline = self._load_diarization_pipeline()

            if self._diarization_pipeline is None:
                logger.warning("  Diarization pipeline could not be loaded.")
                return aligned_result, False

            audio = whisperx.load_audio(str(audio_path))
            diarize_segments = self._diarization_pipeline(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

            result_with_speakers = whisperx.assign_word_speakers(
                diarize_segments, aligned_result,
            )

            speakers = set()
            for seg in result_with_speakers.get("segments", []):
                spk = seg.get("speaker")
                if spk:
                    speakers.add(spk)

            logger.info("  Diarization done | speakers=%d", len(speakers))
            return result_with_speakers, True

        except Exception as exc:
            logger.warning("  Diarization failed, single-speaker fallback: %s", exc)
            return aligned_result, False

    def _load_diarization_pipeline(self) -> Any:
        """
        BUG-4 FIX : try multiple parameter names for HF auth token.

        WhisperX DiarizationPipeline changed its __init__ signature:
        - Old versions: use_auth_token=
        - New versions (>=3.7, pyannote >=4.0): token=

        We try both to maintain compatibility.
        """
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            logger.warning("  whisperx.diarize not available")
            return None

        import inspect
        init_params = inspect.signature(DiarizationPipeline.__init__).parameters

        # ── Strategy 1 : inspect the signature to pick the right kwarg ──
        if "token" in init_params:
            try:
                logger.info("  Loading DiarizationPipeline(token=...)")
                pipeline = DiarizationPipeline(
                    token=self.hf_token,
                    device=self.device,
                )
                logger.info("  ✅ DiarizationPipeline loaded with token=")
                return pipeline
            except Exception as exc:
                logger.debug("  DiarizationPipeline(token=) failed: %s", exc)

        if "use_auth_token" in init_params:
            try:
                logger.info("  Loading DiarizationPipeline(use_auth_token=...)")
                pipeline = DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device,
                )
                logger.info("  ✅ DiarizationPipeline loaded with use_auth_token=")
                return pipeline
            except Exception as exc:
                logger.debug("  DiarizationPipeline(use_auth_token=) failed: %s", exc)

        # ── Strategy 2 : brute force try both ───────────────────────────
        # (in case inspect doesn't catch **kwargs)
        for kwarg_name in ("token", "use_auth_token"):
            try:
                logger.info("  Trying DiarizationPipeline(%s=...)", kwarg_name)
                pipeline = DiarizationPipeline(
                    **{kwarg_name: self.hf_token},
                    device=self.device,
                )
                logger.info("  ✅ DiarizationPipeline loaded with %s=", kwarg_name)
                return pipeline
            except TypeError:
                # Wrong parameter name → TypeError
                continue
            except Exception as exc:
                logger.debug("  DiarizationPipeline(%s=) failed: %s", kwarg_name, exc)
                continue

        logger.warning(
            "  ⚠️ Could not load DiarizationPipeline with any auth parameter. "
            "Check your WhisperX/pyannote version and HF token."
        )
        return None

    # ==================================================================
    # 6. Merge transcript
    # ==================================================================

    def _merge_transcript(
        self, whisper_result: Dict[str, Any], file_language: str,
    ) -> List[TranscriptSegment]:
        raw_segments = whisper_result.get("segments", []) or []
        output: List[TranscriptSegment] = []
        for seg in raw_segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start_time = float(seg.get("start", seg.get("start_time", 0.0)) or 0.0)
            end_time = float(seg.get("end", seg.get("end_time", start_time)) or start_time)
            language = str(seg.get("language", file_language) or file_language)
            words = seg.get("words") or []
            confidence = self._estimate_segment_confidence(seg)
            speaker = str(seg.get("speaker", "SPEAKER_00") or "SPEAKER_00")
            output.append(TranscriptSegment(
                text=text, start_time=start_time, end_time=end_time,
                language=language, speaker=speaker, confidence=confidence, words=words,
            ))
        return output

    # ==================================================================
    # 7. Language detection + fix
    # ==================================================================

    def _detect_and_fix_languages(
        self, segments: List[TranscriptSegment], file_language: str,
    ) -> List[TranscriptSegment]:
        detector = _get_language_detector()
        for seg in segments:
            if not seg.language or seg.language == "unknown":
                seg.language = file_language
            if detector and len(seg.text.split()) >= 5:
                detected = _detect_segment_language(seg.text, detector)
                if detected and detected != "unknown":
                    seg.language = detected
        return segments

    # ==================================================================
    # 8. Confidence filter (BUG-1 FIX)
    # ==================================================================

    def _is_hallucination(self, text: str) -> bool:
        """Detect Whisper music/silence hallucinations that pass the confidence
        filter (they often come back HIGH confidence). Catches:
          - looping repeated phrases ("thank you. thank you. thank you.")
          - known boilerplate ("subtitles by", "thanks for watching")
          - a single short token repeated many times
        """
        t = (text or "").strip().lower()
        if not t:
            return True
        # known boilerplate hallucinations
        BOILERPLATE = (
            "subtitles by", "amara.org", "thanks for watching",
            "thank you for watching", "please subscribe", "subscribe to",
            "transcribed by", "subtitles created by",
        )
        if any(b in t for b in BOILERPLATE):
            return True
        words = t.split()
        if len(words) >= 4:
            # repetition ratio: unique words / total. Loops have very low ratio.
            uniq = len(set(words))
            if uniq / len(words) < 0.35:
                return True
            # same phrase repeated: split into halves, compare
            half = len(words) // 2
            if words[:half] == words[half:half * 2] and half >= 2:
                return True
        return False

    def _confidence_filter(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        filtered = []
        dropped_halluc = 0
        for seg in segments:
            wc = len(seg.text.split())
            conf = seg.confidence
            # hallucination guard FIRST — catches high-confidence music garbage
            if self._is_hallucination(seg.text):
                dropped_halluc += 1
                continue
            if wc >= MIN_WORDS and conf >= MIN_CONFIDENCE:
                filtered.append(seg)
            elif wc >= MIN_WORDS_LONG and conf >= MIN_CONFIDENCE_LONG:
                filtered.append(seg)
            else:
                logger.debug("  Dropped: conf=%.2f words=%d text='%s'", conf, wc, seg.text[:60])
        if dropped_halluc:
            logger.info("  Hallucination guard: dropped %d music/silence segments", dropped_halluc)
        return filtered

    # ==================================================================
    # Confidence estimation
    # ==================================================================

    def _estimate_segment_confidence(self, seg: Dict[str, Any]) -> float:
        if "confidence" in seg:
            try:
                return max(0.0, min(1.0, float(seg["confidence"])))
            except Exception:
                pass
        words = seg.get("words", [])
        if words:
            scores = []
            for w in words:
                score = w.get("score") or w.get("confidence") or w.get("probability")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (TypeError, ValueError):
                        pass
            if scores:
                return sum(scores) / len(scores)
        avg_logprob = seg.get("avg_logprob")
        if avg_logprob is not None:
            try:
                val = float(avg_logprob)
                if val >= -0.2: return 0.95
                if val >= -0.5: return 0.88
                if val >= -1.0: return 0.78
                if val >= -1.5: return 0.66
                return 0.50
            except (TypeError, ValueError):
                pass
        return 0.75

    # ==================================================================
    # Language distribution
    # ==================================================================

    def _compute_language_distribution(
        self, fallback_language: str, segments: List[TranscriptSegment],
    ) -> Tuple[str, Dict[str, float]]:
        if not segments:
            lang = fallback_language or "unknown"
            return lang, {lang: 1.0}
        duration_per_lang: Dict[str, float] = {}
        total_duration = 0.0
        for seg in segments:
            lang = seg.language or fallback_language or "unknown"
            if lang == "unknown":
                lang = fallback_language or "unknown"
            dur = max(0.0, seg.end_time - seg.start_time)
            duration_per_lang[lang] = duration_per_lang.get(lang, 0.0) + dur
            total_duration += dur
        if total_duration <= 0.0:
            dominant = fallback_language or "unknown"
            return dominant, {dominant: 1.0}
        distribution = {lang: round(dur / total_duration, 4) for lang, dur in duration_per_lang.items()}
        dominant = max(distribution, key=distribution.get)
        return dominant, distribution

    # ==================================================================
    # Duration
    # ==================================================================

    def _compute_duration(self, segments: List[TranscriptSegment]) -> float:
        if not segments:
            return 0.0
        return max(float(seg.end_time) for seg in segments)