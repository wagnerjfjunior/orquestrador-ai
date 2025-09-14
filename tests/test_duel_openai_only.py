# =============================================================================
# File: tests/test_duel_openai_only.py
# Version: 2025-09-14 16:45:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: A função de mock `_fake_provider_call` foi convertida para `async def`.
# - CORREÇÃO: O mock de `judge_answers` também precisa ser `async`.
# - O teste agora espera 200 OK, pois o duelo deve funcionar com apenas 1 provedor.
# =============================================================================
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_openai_only_ok(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    async def _fake_provider_call(name, prompt): # <-- MUDANÇA: async def
        if name == "openai":
            return {"provider": "openai", "answer": "Paris é a capital da França."}
        raise RuntimeError("Gemini not configured")
    
    async def fake_judge(q, a, b): # <-- MUDANÇA: async def
        return {"winner": "a", "reason": "A is valid"}

    monkeypatch.setattr(m, "_provider_call", _fake_provider_call)
    monkeypatch.setattr(m, "judge_answers", fake_judge)

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "duel"
    assert body["responses"]["openai"]["ok"] is True
    assert "Paris" in (body["responses"]["openai"]["answer"] or "")
    assert body["verdict"]["winner"] in ("openai", "tie")
