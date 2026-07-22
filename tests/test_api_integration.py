"""
Integration tests for the FastAPI application.
Uses TestClient to test endpoints without a running server.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client. Components may not be fully loaded in test env."""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "components" in data

    def test_health_has_all_component_keys(self, client):
        response = client.get("/health")
        components = response.json()["components"]
        expected_keys = {"store", "transcriber", "embedder", "chunker", "reranker", "vision", "llm"}
        assert expected_keys == set(components.keys())


class TestAppEndpoint:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_app_returns_html(self, client):
        response = client.get("/app")
        assert response.status_code == 200


class TestVideosEndpoint:
    def test_videos_list(self, client):
        response = client.get("/videos")
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        assert "page" in data

    def test_videos_pagination(self, client):
        response = client.get("/videos?page=1&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1


class TestSearchEndpoint:
    def test_search_with_empty_store(self, client):
        response = client.post("/search", json={
            "query": "test query",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert isinstance(data["results"], list)


class TestAskEndpoint:
    def test_ask_with_empty_store(self, client):
        response = client.post("/ask", json={
            "query": "What is discussed in the video?",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "answer" in data

    def test_ask_with_arabic_query(self, client):
        response = client.post("/ask", json={
            "query": "ما هو موضوع الفيديو؟",
            "top_k": 3,
            "language": "ar",
        })
        assert response.status_code == 200


class TestStatsEndpoint:
    def test_stats(self, client):
        response = client.get("/stats")
        assert response.status_code == 200


class TestVideoNotFound:
    def test_video_file_not_found(self, client):
        response = client.get("/video-file/nonexistent_video_12345")
        assert response.status_code == 404
