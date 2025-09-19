# app/metrics.py
from typing import Optional

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Contador de requisições do /ask por provider e status (success|error)
ASK_REQUESTS_TOTAL = Counter(
    "ask_requests_total",
    "Total de chamadas ao /ask",
    labelnames=("provider", "status"),
)

# Latência do /ask por provider e status (em segundos)
ASK_LATENCY_SECONDS = Histogram(
    "ask_latency_seconds",
    "Latência das chamadas ao /ask (s)",
    labelnames=("provider", "status"),
)


def setup_metrics(app, endpoint: str = "/metrics"):
    """
    Instrumenta a app FastAPI e expõe /metrics.
    """
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint=endpoint)


def record_ask(provider: str, status: str, duration_ms: Optional[float] = None) -> None:
    """
    Registra uma ocorrência do /ask nas métricas personalizadas.
    - provider: "echo" | "openai" | "gemini" | ...
    - status: "success" | "error" (ou outro rótulo que desejar padronizar)
    - duration_ms: opcional; se fornecido, registra no histograma em segundos
    """
    p = (provider or "unknown").lower()
    s = (status or "unknown").lower()

    # incrementa contador
    ASK_REQUESTS_TOTAL.labels(provider=p, status=s).inc()

    # observa latência se fornecida
    if duration_ms is not None:
        ASK_LATENCY_SECONDS.labels(provider=p, status=s).observe(duration_ms / 1000.0)
