# tests/test_observability.py
from starlette.testclient import TestClient
from app.main import app

def test_trace_middleware_logs_request_start_and_end(capsys):
    """
    Garante que o TraceMiddleware loga 'request.start' e 'request.end'
    em formato JSON no stdout durante uma requisição.
    """
    client = TestClient(app)

    # limpa buffer
    capsys.readouterr()

    # faz uma request simples
    r = client.get("/health")
    assert r.status_code == 200

    # captura stdout
    captured = capsys.readouterr()
    out_lines = captured.out.splitlines()

    # deve conter eventos de início e fim
    has_start = any('"event": "request.start"' in line or '"request.start"' in line for line in out_lines)
    has_end = any('"event": "request.end"' in line or '"request.end"' in line for line in out_lines)
    assert has_start, f"Não encontrou log 'request.start' em: {captured.out}"
    assert has_end, f"Não encontrou log 'request.end' em: {captured.out}"

    # deve conter um request_id nos logs
    # (ambos logs têm request_id — validamos pelo menos um)
    has_request_id = any('"request_id"' in line for line in out_lines)
    assert has_request_id, f"Não encontrou 'request_id' nos logs: {captured.out}"

def test_trace_middleware_includes_path_and_method(capsys):
    """
    Verifica se os logs incluem path e method (metadados básicos).
    """
    client = TestClient(app)
    capsys.readouterr()

    r = client.get("/ready")
    assert r.status_code == 200

    captured = capsys.readouterr()
    out = captured.out

    assert '"path":' in out, f"Log não contém 'path': {out}"
    assert '"method": "GET"' in out, f"Log não contém 'method': {out}"
