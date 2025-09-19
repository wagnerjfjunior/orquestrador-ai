# =============================================================================
# File: tests/test_fallback.py
# Version: 2025-09-14 16:30:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: As funções de mock `boom` e `ok` foram convertidas para `async def`.
# - CORREÇÃO: `test_fallback_todos_falham` agora espera a mensagem de erro correta.
# - As funções de mock dos provedores agora precisam ser assíncronas.
# =============================================================================
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_fallback_openai_falha_e_gemini_sucesso(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Erro simulado no OpenAI")

    async def ok(prompt):  # <-- MUDANÇA: async def
        return {"provider": "gemini", "model": "gemini-1.5-flash", "answer": "ok gemini", "usage": {}}

    monkeypatch.setattr("app.main.ask_openai", boom)
    monkeypatch.setattr("app.main.ask_gemini", ok)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "ok gemini"


def test_fallback_provider_explicito_sem_fallback(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Erro simulado no OpenAI")

    monkeypatch.setattr("app.main.ask_openai", boom)

    r = client.post("/ask?provider=openai&use_fallback=false", json={"prompt": "hi"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower()


def test_fallback_todos_falham(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 503
    # CORREÇÃO: O erro final na cadeia de fallback é o do último provedor (gemini)
    assert "gemini_api_key não configurada" in r.json()["detail"].lower()
