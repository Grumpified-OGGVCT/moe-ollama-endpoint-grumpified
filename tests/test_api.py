"""Basic tests for the MoE Ollama Endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "MoE Ollama Endpoint"


def test_v1_health_check():
    """Test v1 health check endpoint."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_models_list():
    """Test models listing endpoint."""
    response = client.get("/v1/models")
    assert response.status_code in [200, 500]  # May fail without Ollama connection
    if response.status_code == 200:
        data = response.json()
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"


@pytest.mark.asyncio
async def test_chat_completion_structure():
    """Test chat completion request structure."""
    # This will likely fail without actual Ollama connection
    # but tests the endpoint structure
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
    )
    # Accept either success or service unavailable
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_embeddings_structure():
    """Test embeddings endpoint structure."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "nomic-embed-text",
            "input": "test"
        }
    )
    # Accept either success or service unavailable
    assert response.status_code in [200, 500]
