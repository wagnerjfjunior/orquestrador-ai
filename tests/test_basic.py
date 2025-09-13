from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "live"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ask_echo():
    r = client.post("/ask?provider=echo", json={"prompt": "ping"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "echo"
    assert data["answer"] == "ping"
