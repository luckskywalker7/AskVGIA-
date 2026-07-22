"""
Tests for core.storage (Qdrant integration)
These tests use a temporary Qdrant local path to avoid conflicts.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestVideoStoreInit:
    def test_store_creates_collection(self):
        """Test that VideoStore can initialize with a temp path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("config.settings.settings") as mock_settings:
                mock_settings.qdrant_mode = "local"
                mock_settings.qdrant_path = tmpdir
                mock_settings.qdrant_host = "localhost"
                mock_settings.qdrant_port = 6333
                mock_settings.qdrant_collection = "test_collection"
                mock_settings.qdrant_timeout = 10.0
                mock_settings.qdrant_prefer_grpc = False
                mock_settings.nomic_dense_dim = 768
                mock_settings.bge_dense_dim = 1024
                mock_settings.rrf_k = 60
                mock_settings.weight_nomic = 0.20
                mock_settings.weight_bge_dense = 0.50
                mock_settings.weight_bge_sparse = 0.30

                from core.storage import VideoStore
                store = VideoStore()
                assert store.collection_name == "test_collection"

    def test_store_list_videos_empty(self):
        """Test that an empty store returns no videos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("config.settings.settings") as mock_settings:
                mock_settings.qdrant_mode = "local"
                mock_settings.qdrant_path = tmpdir
                mock_settings.qdrant_host = "localhost"
                mock_settings.qdrant_port = 6333
                mock_settings.qdrant_collection = "test_empty"
                mock_settings.qdrant_timeout = 10.0
                mock_settings.qdrant_prefer_grpc = False
                mock_settings.nomic_dense_dim = 768
                mock_settings.bge_dense_dim = 1024
                mock_settings.rrf_k = 60
                mock_settings.weight_nomic = 0.20
                mock_settings.weight_bge_dense = 0.50
                mock_settings.weight_bge_sparse = 0.30

                from core.storage import VideoStore
                store = VideoStore()
                videos = store.list_videos()
                assert videos == []
