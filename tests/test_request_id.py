# tests/test_request_id.py
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_response_contains_request_id_header():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]  # não vazio


def test_request_id_is_propagated_from_request_to_response():
    custom = "abc123xyz"
    r = client.get("/ready", headers={"x-request-id": custom})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == custom
