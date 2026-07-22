"""
Tests for core.contracts
"""
import pytest
from core.contracts import (
    SearchResult,
    SemanticChunk,
    TranscriptSegment,
    NormalizedText,
    LanguageDetection,
)


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(
            chunk_id="abc123",
            video_id="vid_01",
            text="Some text here",
            score=0.85,
            start_time=10.0,
            end_time=15.0,
            language="en",
            chunk_type="speech",
        )
        assert r.chunk_id == "abc123"
        assert r.score == 0.85
        assert r.chunk_type == "speech"

    def test_to_dict(self):
        r = SearchResult(
            chunk_id="abc",
            video_id="vid",
            text="test",
            score=0.5,
        )
        d = r.to_dict()
        assert d["chunk_id"] == "abc"
        assert d["score"] == 0.5
        assert "chunk_type" in d
        assert "strategy" in d

    def test_timestamp_formatted(self):
        r = SearchResult(
            chunk_id="a",
            video_id="v",
            text="t",
            score=0.1,
            start_fmt="01:23",
            end_fmt="01:45",
        )
        assert r.timestamp_formatted == "01:23-01:45"

    def test_timestamp_formatted_no_end(self):
        r = SearchResult(
            chunk_id="a",
            video_id="v",
            text="t",
            score=0.1,
            start_fmt="02:00",
        )
        assert r.timestamp_formatted == "02:00"

    def test_default_strategy(self):
        r = SearchResult(chunk_id="a", video_id="v", text="t", score=0.0)
        assert r.strategy == "hybrid"

    def test_strategy_mutable(self):
        r = SearchResult(chunk_id="a", video_id="v", text="t", score=0.0)
        r.strategy = "dense+rerank"
        assert r.strategy == "dense+rerank"


class TestSemanticChunk:
    def test_creation(self):
        c = SemanticChunk(
            chunk_id="ch1",
            video_id="vid1",
            text="Hello world",
            original_text="Hello world",
            normalized_text="hello world",
            search_text="hello world",
            start_time=0.0,
            end_time=5.0,
            language="en",
            chunk_type="speech",
        )
        assert c.duration == 5.0
        assert c.chunk_type == "speech"

    def test_visual_chunk(self):
        c = SemanticChunk(
            chunk_id="vis1",
            video_id="vid1",
            text="Slide: Hybrid Search",
            original_text="Slide: Hybrid Search",
            normalized_text="slide hybrid search",
            search_text="slide hybrid search",
            start_time=10.0,
            end_time=10.0,
            chunk_type="visual",
        )
        assert c.chunk_type == "visual"


class TestTranscriptSegment:
    def test_duration(self):
        seg = TranscriptSegment(text="test", start_time=5.0, end_time=10.0)
        assert seg.duration == 5.0

    def test_zero_duration(self):
        seg = TranscriptSegment(text="test", start_time=5.0, end_time=5.0)
        assert seg.duration == 0.0
