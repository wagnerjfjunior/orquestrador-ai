# tests/test_fallback.py
import pytest
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_fallback_openai_falha_e_gemini_sucesso(monkeypatch):
    # Cadeia padrão no settings costuma ser ["openai", "gemini"] — assumimos isso.
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Erro simulado no OpenAI")

    def ok(prompt):
        return {"provider": "gemini", "model": "gemini-1.5-flash", "answer": "ok gemini", "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}

    monkeypatch.setattr("app.main.ask_openai", lambda prompt: boom(prompt))
    monkeypatch.setattr("app.main.ask_gemini", lambda prompt: ok(prompt))

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "ok gemini"


def test_fallback_provider_explicito_sem_fallback(monkeypatch):
    # Se use_fallback=false, não deve tentar o próximo
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Erro simulado no OpenAI")

    monkeypatch.setattr("app.main.ask_openai", lambda prompt: boom(prompt))

    r = client.post("/ask?provider=openai&use_fallback=false", json={"prompt": "hi"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower() or "simulado" in r.json()["detail"].lower()


def test_fallback_todos_falham(monkeypatch):
    # Nem openai nem gemini disponíveis (ou ambos falham)
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    # Como nenhum está configurado, o último erro é 503 (não configurado)
    assert r.status_code == 503
    assert "não configurado" in r.json()["detail"].lower()
