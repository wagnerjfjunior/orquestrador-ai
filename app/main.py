# app/main.py
from fastapi import FastAPI, Body, HTTPException
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics

app = FastAPI(
    title="orquestrador-ai",
    version="0.1.0",
    description="Orquestrador multi-IA com observabilidade e métricas",
)

# --- Middlewares e métricas ---
app.add_middleware(TraceMiddleware)
setup_metrics(app)

# --- Rotas básicas ---
@app.get("/", tags=["infra"])
def root():
    """Usado no teste: deve retornar {'status': 'live'}"""
    logger.info("root.live")
    return {"status": "live"}

@app.get("/health", tags=["infra"])
def health():
    """Health check simples"""
    logger.info("health.ok")
    return {"status": "ok"}

@app.get("/ready", tags=["infra"])
def readiness():
    """Readiness para probes"""
    logger.info("readiness.ok")
    return {"status": "ready"}

# --- Orquestrador: endpoint /ask ---
@app.post("/ask", tags=["ask"])
def ask(provider: str = "echo", payload: dict = Body(...)):
    """
    Endpoint principal do orquestrador.

    Para os testes:
      - provider=echo deve devolver o próprio prompt em 'answer'
      Exemplo:
        POST /ask?provider=echo
        { "prompt": "ping" }
        -> { "provider": "echo", "answer": "ping" }
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    if provider == "echo":
        logger.info("ask.echo", prompt=prompt)
        # 'answer' é obrigatório para o teste; 'output' mantido por compatibilidade
        return {"provider": "echo", "answer": prompt, "output": prompt}

    # Providers reais (openai, gemini, etc.) entram aqui posteriormente
    raise HTTPException(status_code=400, detail=f"Provider não suportado: {provider}")
