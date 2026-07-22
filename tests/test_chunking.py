"""
Tests for core.chunking
"""
import pytest
from core.chunking import SemanticChunker


@pytest.fixture
def chunker():
    return SemanticChunker()


class TestBasicChunking:
    def test_english_segments_produce_chunks(self, chunker, sample_english_segments):
        result = chunker.chunk(
            {"video_id": "test_en", "language": "en", "segments": sample_english_segments}
        )
        assert result.total_chunks > 0
        assert result.video_id == "test_en"
        assert result.language == "en"

    def test_arabic_segments_produce_chunks(self, chunker, sample_arabic_segments):
        result = chunker.chunk(
            {"video_id": "test_ar", "language": "ar", "segments": sample_arabic_segments}
        )
        assert result.total_chunks > 0
        assert result.language == "ar"

    def test_french_segments_produce_chunks(self, chunker, sample_french_segments):
        result = chunker.chunk(
            {"video_id": "test_fr", "language": "fr", "segments": sample_french_segments}
        )
        assert result.total_chunks > 0

    def test_mixed_language_segments(self, chunker, sample_mixed_segments):
        result = chunker.chunk(
            {"video_id": "test_mix", "segments": sample_mixed_segments}
        )
        assert result.total_chunks > 0


class TestChunkQuality:
    def test_chunk_has_required_fields(self, chunker, sample_english_segments):
        result = chunker.chunk(
            {"video_id": "test", "language": "en", "segments": sample_english_segments}
        )
        chunk = result.chunks[0]
        assert chunk.chunk_id
        assert chunk.video_id == "test"
        assert chunk.text
        assert chunk.original_text
        assert chunk.normalized_text
        assert chunk.search_text
        assert chunk.start_time >= 0
        assert chunk.end_time >= chunk.start_time
        assert chunk.language
        assert chunk.chunk_type == "speech"

    def test_chunk_timestamps_are_ordered(self, chunker, sample_english_segments):
        result = chunker.chunk(
            {"video_id": "test", "language": "en", "segments": sample_english_segments}
        )
        for chunk in result.chunks:
            assert chunk.end_time >= chunk.start_time

    def test_chunk_ids_are_unique(self, chunker, sample_english_segments):
        result = chunker.chunk(
            {"video_id": "test", "language": "en", "segments": sample_english_segments}
        )
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))


class TestEdgeCases:
    def test_empty_segments(self, chunker):
        result = chunker.chunk({"video_id": "empty", "language": "en", "segments": []})
        assert result.total_chunks == 0
        assert result.chunks == []

    def test_none_input(self, chunker):
        result = chunker.chunk(None, video_id="null_test")
        assert result.total_chunks == 0

    def test_segments_with_empty_text(self, chunker, empty_segments):
        result = chunker.chunk(
            {"video_id": "gaps", "language": "en", "segments": empty_segments}
        )
        # Only the valid segment should produce a chunk
        for chunk in result.chunks:
            assert chunk.text.strip()

    def test_single_word_segment(self, chunker):
        result = chunker.chunk({
            "video_id": "tiny",
            "language": "en",
            "segments": [{"text": "Hello", "start": 0.0, "end": 0.5}],
        })
        # Should handle gracefully (may or may not produce a chunk)
        assert isinstance(result.chunks, list)


class TestAliases:
    def test_create_semantic_chunks_alias(self, chunker, sample_english_segments):
        result = chunker.create_semantic_chunks(
            {"video_id": "alias_test", "language": "en", "segments": sample_english_segments}
        )
        assert result.total_chunks > 0

    def test_generate_chunks_alias(self, chunker, sample_english_segments):
        result = chunker.generate_chunks(
            {"video_id": "alias_test2", "language": "en", "segments": sample_english_segments}
        )
        assert result.total_chunks > 0
