# app/metrics.py
from prometheus_fastapi_instrumentator import Instrumentator

def setup_metrics(app):
    """
    Expõe métricas em /metrics e instrumenta rotas automaticamente.
    """
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
