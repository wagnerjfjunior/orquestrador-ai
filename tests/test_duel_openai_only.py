# tests/test_duel_openai_only.py
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_openai_only_ok():
    # Só OpenAI “configurado”
    m.openai_configured = lambda: True
    m.gemini_configured = lambda: False

    # Mock do provider
    def _fake_provider_call(name, prompt):
        assert name == "openai"
        return {"provider": "openai", "answer": "Paris é a capital da França."}

    m._provider_call = _fake_provider_call

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "duel"
    assert body["responses"]["openai"]["ok"] is True
    assert "Paris" in (body["responses"]["openai"]["answer"] or "")
    assert body["verdict"]["winner"] in ("openai", "tie")
