from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Language / normalization contracts
# ----------------------------------------------------------------------

@dataclass
class LanguageDetection:
    """
    Result returned by the language detector.
    """
    code: str
    confidence: float
    source: str
    script: str = "unknown"


@dataclass
class NormalizedText:
    """
    Canonical text representation used by retrieval.

    original_text    -> raw text as spoken / extracted
    normalized_text  -> cleaned language-aware version
    search_text      -> retrieval-oriented string
    """
    original_text: str
    normalized_text: str
    search_text: str
    language: str


# ----------------------------------------------------------------------
# Transcription contracts
# ----------------------------------------------------------------------

@dataclass
class TranscriptWord:
    """
    Optional word-level timing unit.
    """
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class TranscriptSegment:
    """
    Canonical transcript segment used by the chunker.
    """
    text: str
    start_time: float
    end_time: float
    language: str = "unknown"
    speaker: str = "SPEAKER_00"
    confidence: float = 1.0
    words: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


@dataclass
class TranscriptionResult:
    """
    Canonical transcript result returned by the transcriber.
    """
    video_id: str
    segments: List[TranscriptSegment]
    language: str
    language_distribution: Dict[str, float]
    total_segments: int
    duration_seconds: float
    has_diarization: bool
    processing_seconds: float
    audio_path: Optional[str] = None


# ----------------------------------------------------------------------
# Chunking contracts
# ----------------------------------------------------------------------

@dataclass
class SemanticChunk:
    """
    Canonical chunk used by the retrieval layer.

    text             -> text exposed to the user / answer generator
    normalized_text  -> cleaned version
    search_text      -> retrieval-optimized version
    metadata         -> free-form metadata for downstream steps
    """
    chunk_id: str
    video_id: str
    text: str
    original_text: str
    normalized_text: str
    search_text: str
    start_time: float
    end_time: float
    language: str = "unknown"
    speaker: str = "unknown"
    topic: str = "general"
    chunk_type: str = "speech"
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


@dataclass
class ChunkingResult:
    """
    Output of the chunker.
    """
    video_id: str
    chunks: List[SemanticChunk]
    total_segments: int
    total_chunks: int
    language: str
    processing_seconds: float


# ----------------------------------------------------------------------
# Search contracts
# ----------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    Canonical search result used across storage, reranking, API and LLM.
    """
    chunk_id: str
    video_id: str
    text: str
    score: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_fmt: Optional[str] = None
    end_fmt: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    topic: Optional[str] = None
    chunk_type: str = "speech"
    strategy: str = "hybrid"
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    key_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def segment_id(self) -> str:
        return self.chunk_id

    @property
    def original_text(self) -> str:
        return self.text

    @property
    def timestamp_formatted(self) -> str:
        if self.start_fmt and self.end_fmt:
            return f"{self.start_fmt}-{self.end_fmt}"
        if self.start_fmt:
            return self.start_fmt
        return "??:??"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "text": self.text,
            "score": self.score,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_fmt": self.start_fmt,
            "end_fmt": self.end_fmt,
            "language": self.language,
            "speaker": self.speaker,
            "topic": self.topic,
            "chunk_type": self.chunk_type,
            "strategy": self.strategy,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "key_points": self.key_points,
            "metadata": self.metadata,
        }
