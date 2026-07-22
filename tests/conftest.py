"""
Shared pytest fixtures for the project.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List


# ══════════════════════════════════════════════════════════════════════════════
# Sample data fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_english_segments() -> List[Dict[str, Any]]:
    return [
        {"text": "Today we discuss retrieval augmented generation.", "start": 0.0, "end": 4.2, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "The key idea is to ground answers in evidence.", "start": 4.3, "end": 8.1, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "This reduces hallucinations.", "start": 8.2, "end": 10.5, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "Then we combine dense and sparse retrieval.", "start": 11.0, "end": 15.0, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "The reranker refines the final ordering.", "start": 15.5, "end": 19.0, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "Let me show you the evaluation results now.", "start": 20.0, "end": 24.0, "speaker": "SPEAKER_00", "language": "en"},
    ]


@pytest.fixture
def sample_arabic_segments() -> List[Dict[str, Any]]:
    return [
        {"text": "السلام عليكم. اليوم سنتحدث عن البحث الدلالي.", "start": 0.0, "end": 4.5, "speaker": "SPEAKER_00", "language": "ar"},
        {"text": "الفكرة الأساسية هي تأسيس الإجابات على الأدلة.", "start": 4.6, "end": 9.0, "speaker": "SPEAKER_00", "language": "ar"},
        {"text": "هذا يقلل من الهلوسة ويحسن الثقة.", "start": 9.1, "end": 12.5, "speaker": "SPEAKER_00", "language": "ar"},
        {"text": "ثم نجمع بين الاسترجاع الكثيف والمتناثر.", "start": 13.0, "end": 17.0, "speaker": "SPEAKER_00", "language": "ar"},
    ]


@pytest.fixture
def sample_french_segments() -> List[Dict[str, Any]]:
    return [
        {"text": "Bonjour à tous. Aujourd'hui nous parlons de recherche hybride.", "start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "language": "fr"},
        {"text": "L'idée principale est de combiner dense et sparse.", "start": 5.1, "end": 9.5, "speaker": "SPEAKER_00", "language": "fr"},
        {"text": "Le reranking améliore ensuite le classement final.", "start": 10.0, "end": 14.0, "speaker": "SPEAKER_00", "language": "fr"},
    ]


@pytest.fixture
def sample_mixed_segments() -> List[Dict[str, Any]]:
    return [
        {"text": "السلام عليكم.", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "language": "ar"},
        {"text": "Now let me explain dense retrieval.", "start": 2.5, "end": 6.0, "speaker": "SPEAKER_00", "language": "en"},
        {"text": "وأخيراً سنعود إلى التقييم.", "start": 6.5, "end": 9.0, "speaker": "SPEAKER_00", "language": "ar"},
    ]


@pytest.fixture
def sample_transcription_dict(sample_english_segments) -> Dict[str, Any]:
    return {
        "video_id": "test_video_01",
        "language": "en",
        "segments": sample_english_segments,
    }


@pytest.fixture
def empty_segments() -> List[Dict[str, Any]]:
    return [
        {"text": "", "start": 0.0, "end": 1.0},
        {"text": "  ", "start": 1.0, "end": 2.0},
        {"text": "Valid text here.", "start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
