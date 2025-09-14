# tests/test_ask_providers.py
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_openai_success(monkeypatch):
    # finge que a chave está configurada
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    # mock da chamada ao cliente openai
    def fake_ask_openai(prompt):
        assert prompt == "olá openai"
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "answer": "oi, daqui é o openai",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("app.main.ask_openai", fake_ask_openai)

    r = client.post("/ask?provider=openai", json={"prompt": "olá openai"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "openai"
    assert data["answer"] == "oi, daqui é o openai"
    assert "usage" in data


def test_ask_gemini_success(monkeypatch):
    # finge que a chave está configurada
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    # mock da chamada ao cliente gemini
    def fake_ask_gemini(prompt):
        assert prompt == "olá gemini"
        return {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "answer": "oi, daqui é o gemini",
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }

    monkeypatch.setattr("app.main.ask_gemini", fake_ask_gemini)

    r = client.post("/ask?provider=gemini", json={"prompt": "olá gemini"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "oi, daqui é o gemini"
    assert "usage" in data


def test_ask_openai_not_configured(monkeypatch):
    # força "não configurado" → deve retornar 503
    monkeypatch.setattr("app.main.openai_configured", lambda: False)

    r = client.post("/ask?provider=openai", json={"prompt": "qualquer"})
    assert r.status_code == 503
    assert "não configurada" in r.json()["detail"].lower()


def test_ask_gemini_provider_error(monkeypatch):
    # configurado, mas cliente lança erro → deve virar 502
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Rate limit atingido na OpenAI. Tente novamente mais tarde.")  # exemplo de msg

    monkeypatch.setattr("app.main.ask_gemini", lambda prompt: boom(prompt))

    r = client.post("/ask?provider=gemini", json={"prompt": "teste"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower() or "limit" in r.json()["detail"].lower()
