import pytest
from fastapi.testclient import TestClient

import backend.app.dependencies as dependencies
import backend.app.main as main
from backend.app.main import app


class FakeChatter:
    def __init__(self):
        self.calls = []

    def chat(self, message: str) -> str:
        self.calls.append(message)
        return f"echo: {message}"


@pytest.fixture(autouse=True)
def reset_chatter_cache():
    if hasattr(dependencies.get_chatter, "cache_clear"):
        dependencies.get_chatter.cache_clear()
    yield
    if hasattr(dependencies.get_chatter, "cache_clear"):
        dependencies.get_chatter.cache_clear()


@pytest.fixture
def fake_chatter():
    return FakeChatter()


@pytest.fixture
def client(monkeypatch, fake_chatter):
    monkeypatch.setattr(main, "get_chatter", lambda: fake_chatter)
    app.dependency_overrides[dependencies.get_chatter] = lambda: fake_chatter
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_raise(monkeypatch, fake_chatter):
    """TestClient that doesn't raise exceptions, allowing testing of 500 responses."""
    monkeypatch.setattr(main, "get_chatter", lambda: fake_chatter)
    app.dependency_overrides[dependencies.get_chatter] = lambda: fake_chatter
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "PersistentDM API"}


def test_chat(client):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "echo: Hello"


def test_chat_requires_message(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["loc"] == ["body", "message"]
    assert "required" in error_detail["msg"].lower()


def test_chat_rejects_non_string_message(client):
    """Test that message must be a string, not a number or object."""
    response = client.post("/chat", json={"message": 123})
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["loc"] == ["body", "message"]
    assert (
        "string_type" in error_detail["type"].lower()
        or "str" in error_detail["msg"].lower()
    )


def test_chat_rejects_message_as_object(client):
    """Test that message cannot be an object."""
    response = client.post("/chat", json={"message": {"text": "Hello"}})
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["loc"] == ["body", "message"]


def test_chat_rejects_invalid_json(client):
    """Test that invalid JSON returns 422."""
    response = client.post(
        "/chat", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_chat_rejects_missing_content_type(client, fake_chatter):
    """Test that missing Content-Type header is handled. FastAPI is lenient and will parse valid JSON even without Content-Type."""
    response = client.post("/chat", content='{"message": "Hello"}')
    # FastAPI accepts valid JSON even without Content-Type header
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "echo: Hello"
    assert fake_chatter.calls == ["Hello"]


def test_chat_with_empty_string(client, fake_chatter):
    """Test that empty string message is accepted (Pydantic allows it by default)."""
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "echo: "
    assert fake_chatter.calls == [""]


def test_chat_with_whitespace_only(client, fake_chatter):
    """Test messages with only whitespace."""
    response = client.post("/chat", json={"message": "   \n\t  "})
    assert response.status_code == 200
    assert fake_chatter.calls == ["   \n\t  "]


def test_chat_with_unicode(client, fake_chatter):
    """Test messages with unicode characters."""
    message = "Hello 世界 🌍 emoji"
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == f"echo: {message}"
    assert fake_chatter.calls == [message]


def test_chat_with_long_message(client, fake_chatter):
    """Test messages that are very long."""
    long_message = "A" * 10000
    response = client.post("/chat", json={"message": long_message})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == f"echo: {long_message}"
    assert fake_chatter.calls == [long_message]


def test_chat_with_special_characters(client, fake_chatter):
    """Test messages with special characters."""
    message = '!@#$%^&*()_+-=[]{}|;:"<>,.?/~`'
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200
    assert fake_chatter.calls == [message]


def test_chat_tracks_calls(client, fake_chatter):
    """Test that FakeChatter tracks all chat calls."""
    client.post("/chat", json={"message": "First"})
    client.post("/chat", json={"message": "Second"})
    client.post("/chat", json={"message": "Third"})

    assert len(fake_chatter.calls) == 3
    assert fake_chatter.calls == ["First", "Second", "Third"]


def test_chat_response_format(client):
    """Test that response follows the ChatResponse model structure."""
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert "context" in data
    # context can be None or a string
    assert data["context"] is None or isinstance(data["context"], str)


def test_chat_response_content_type(client):
    """Test that response has correct Content-Type header."""
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_chat_rejects_get_method(client):
    """Test that GET method is not allowed on /chat endpoint."""
    response = client.get("/chat")
    assert response.status_code == 405  # Method Not Allowed


def test_chat_rejects_put_method(client):
    """Test that PUT method is not allowed on /chat endpoint."""
    response = client.put("/chat", json={"message": "Hello"})
    assert response.status_code == 405


def test_chat_rejects_delete_method(client):
    """Test that DELETE method is not allowed on /chat endpoint."""
    response = client.delete("/chat")
    assert response.status_code == 405


def test_chat_handles_chatter_exception(client_no_raise, fake_chatter):
    """Test that exceptions from chatter are properly handled."""

    class FailingChatter:
        def chat(self, message: str) -> str:
            raise RuntimeError("Model failed to generate response")

    failing_chatter = FailingChatter()
    app.dependency_overrides[dependencies.get_chatter] = lambda: failing_chatter

    try:
        response = client_no_raise.post("/chat", json={"message": "Hello"})
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert error_data["detail"]["error"] == "Internal server error"
        assert "Model failed to generate response" in error_data["detail"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_chat_handles_chatter_value_error(client_no_raise, fake_chatter):
    """Test that ValueError from chatter is handled."""

    class ValueErrorChatter:
        def chat(self, message: str) -> str:
            raise ValueError("Invalid input")

    error_chatter = ValueErrorChatter()
    app.dependency_overrides[dependencies.get_chatter] = lambda: error_chatter

    try:
        response = client_no_raise.post("/chat", json={"message": "Hello"})
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert error_data["detail"]["error"] == "Internal server error"
        assert "Invalid input" in error_data["detail"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_health_rejects_post_method(client):
    """Test that POST method is not allowed on /health endpoint."""
    response = client.post("/health")
    assert response.status_code == 405


def test_health_response_content_type(client):
    """Test that /health response has correct Content-Type."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
