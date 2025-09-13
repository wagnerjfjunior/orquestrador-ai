# app/main.py
from fastapi import FastAPI
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="orquestrador-ai",
    version="0.1.0",
    description="Orquestrador multi-IA com observabilidade e métricas"
)

# --- Middlewares e métricas ---
# Observabilidade (logs estruturados + tracing simples)
app.add_middleware(TraceMiddleware)

# Prometheus /metrics
setup_metrics(app)


# --- Rotas básicas ---
@app.get("/health", tags=["infra"])
def health():
    """Checagem simples de saúde da aplicação"""
    logger.info("health.ok")
    return {"status": "ok"}


@app.get("/ready", tags=["infra"])
def readiness():
    """
    Endpoint de readiness (pode ser usado pelo Kubernetes para saber
    se a app está pronta para receber tráfego).
    """
    logger.info("readiness.ok")
    return {"status": "ready"}


@app.get("/", tags=["root"])
def root():
    """Rota raiz para teste rápido"""
    logger.info("root.accessed")
    return {"message": "Bem-vindo ao orquestrador-ai 🚀"}
