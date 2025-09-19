# =============================================================================
# File: tests/test_ask_providers.py
# Version: 2025-09-14 16:45:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: As funções de mock `fake_ask_...` foram convertidas para `async def`.
# - Isso é necessário para que o monkeypatch funcione com o novo código assíncrono.
# =============================================================================
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_openai_success(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    async def fake_ask_openai(prompt):  # <-- MUDANÇA: async def
        assert prompt == "olá openai"
        return {
            "provider": "openai", "model": "gpt-4o-mini", "answer": "oi, daqui é o openai",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("app.main.ask_openai", fake_ask_openai)

    r = client.post("/ask?provider=openai", json={"prompt": "olá openai"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "openai"
    assert data["answer"] == "oi, daqui é o openai"


def test_ask_gemini_success(monkeypatch):
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def fake_ask_gemini(prompt):  # <-- MUDANÇA: async def
        assert prompt == "olá gemini"
        return {
            "provider": "gemini", "model": "gemini-1.5-flash", "answer": "oi, daqui é o gemini",
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }

    monkeypatch.setattr("app.main.ask_gemini", fake_ask_gemini)

    r = client.post("/ask?provider=gemini", json={"prompt": "olá gemini"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "oi, daqui é o gemini"


def test_ask_openai_not_configured(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    r = client.post("/ask?provider=openai", json={"prompt": "qualquer"})
    assert r.status_code == 503
    assert "não configurada" in r.json()["detail"].lower()


def test_ask_gemini_provider_error(monkeypatch):
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Rate limit atingido")

    monkeypatch.setattr("app.main.ask_gemini", boom)

    r = client.post("/ask?provider=gemini", json={"prompt": "teste"})
    assert r.status_code == 502
    assert "limit" in r.json()["detail"].lower()
