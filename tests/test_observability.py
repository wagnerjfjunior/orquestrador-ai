# tests/test_observability.py
from starlette.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from app.main import app
from app.observability import logger, TraceMiddleware, RequestIDMiddleware


client = TestClient(app)


def test_logger_is_configured_and_usable():
    # Deve existir e ter método .info (logger estruturado configurado)
    assert logger is not None
    assert hasattr(logger, "info")


def test_middlewares_exist_and_are_valid_classes():
    # Ambas as classes devem ser middlewares Starlette válidos
    assert issubclass(TraceMiddleware, BaseHTTPMiddleware)
    assert issubclass(RequestIDMiddleware, BaseHTTPMiddleware)


def test_x_request_id_present_on_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]  # não vazio


def test_x_request_id_propagation_on_ready():
    custom = "req-observability-123"
    r = client.get("/ready", headers={"x-request-id": custom})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == custom
