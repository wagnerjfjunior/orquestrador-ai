# tests/test_observability.py
import logging
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_trace_middleware_logs_request_start_and_end(caplog):
    """
    Garante que o TraceMiddleware loga 'request.start' e 'request.end'
    em formato JSON via logging.
    """
    caplog.set_level(logging.INFO)

    r = client.get("/health")
    assert r.status_code == 200

    log_text = caplog.text
    assert "request.start" in log_text, f"Não encontrou log 'request.start'. Logs:\n{log_text}"
    assert "request.end" in log_text, f"Não encontrou log 'request.end'. Logs:\n{log_text}"
    assert '"request_id"' in log_text or "request_id" in log_text, f"Não encontrou 'request_id'. Logs:\n{log_text}"

def test_trace_middleware_includes_path_and_method(caplog):
    """
    Verifica se os logs incluem path e method.
    """
    caplog.set_level(logging.INFO)

    r = client.get("/ready")
    assert r.status_code == 200

    log_text = caplog.text
    assert '"path":' in log_text or '"path"' in log_text, f"Log não contém 'path'. Logs:\n{log_text}"
    assert '"method": "GET"' in log_text or '"method":"GET"' in log_text, f"Log não contém 'method: GET'. Logs:\n{log_text}"
