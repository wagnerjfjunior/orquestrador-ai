# tests/test_metrics_error_counter.py
import re

from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

_METRIC_NAME = r"ask_requests_total"
# Ex.: ask_requests_total{provider="openai",status="error"} 3
_PATTERN = re.compile(
    rf'^{_METRIC_NAME}\{{provider="openai",status="error"\}}\s+([0-9]+(?:\.[0-9]+)?)\s*$'
)

def _scrape_error_counter() -> float:
    r = client.get("/metrics")
    assert r.status_code == 200
    for line in r.text.splitlines():
        m = _PATTERN.match(line.strip())
        if m:
            return float(m.group(1))
    # Se nunca apareceu, considere zero
    return 0.0


def test_error_counter_increments_on_openai_503():
    # 1) Lê valor atual do contador de erros do openai
    before = _scrape_error_counter()

    # 2) Dispara um 503: provider=openai sem OPENAI_API_KEY
    payload = {"prompt": "ping"}
    r = client.post("/ask?provider=openai", json=payload)
    assert r.status_code == 503

    # 3) Lê novamente e valida incremento de +1
    after = _scrape_error_counter()
    assert after == before + 1, f"Esperava {before}+1, obtive {after}"
