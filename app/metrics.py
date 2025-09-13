from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI

def setup_metrics(app: FastAPI) -> None:
    """
    Monta /metrics e coleta métricas de latência, contadores por rota/status, etc.
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
