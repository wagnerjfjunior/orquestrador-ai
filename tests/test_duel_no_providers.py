# tests/test_duel_no_providers.py
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_returns_502_when_no_providers():
    # Nenhum provider “configurado”
    m.openai_configured = lambda: False
    m.gemini_configured = lambda: False

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["detail"]["mode"] == "duel"
    assert body["detail"]["verdict"]["winner"] == "none"
    assert "openai" in body["detail"]["responses"]
    assert "gemini" in body["detail"]["responses"]
