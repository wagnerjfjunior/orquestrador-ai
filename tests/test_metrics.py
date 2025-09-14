# tests/test_metrics.py
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_metrics_exposes_ask_counters():
    # 1) gera uma chamada de sucesso
    r = client.post("/ask?provider=echo", json={"prompt": "ping"})
    assert r.status_code == 200

    # 2) lê /metrics e checa nosso contador customizado
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text

    # Deve ter pelo menos um incremento de sucesso para echo
    # Linha esperada (exemplo):
    # ask_requests_total{provider="echo",status="success"} 1.0
    assert 'ask_requests_total{provider="echo",status="success"}' in text, f"Contador não encontrado em /metrics:\n{text}"
