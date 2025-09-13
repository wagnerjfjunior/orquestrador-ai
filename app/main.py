# app/main.py
from fastapi import FastAPI, Body, HTTPException
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics
from app.config import settings
from app.openai_client import ask_openai, is_configured as openai_configured
from app.gemini_client import ask_gemini, is_configured as gemini_configured

app = FastAPI(
    title="orquestrador-ai",
    version=settings.APP_VERSION,
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

    Exemplos:
      - provider=echo
        { "prompt": "ping" } -> { "provider": "echo", "answer": "ping" }

      - provider=openai
        { "prompt": "Olá!" } -> { "provider": "openai", "model": "...", "answer": "...", "usage": {...} }

      - provider=gemini
        { "prompt": "Olá!" } -> { "provider": "gemini", "model": "...", "answer": "...", "usage": {...} }
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    provider = (provider or "").lower()

    # ECHO (para testes/diagnóstico)
    if provider == "echo":
        logger.info("ask.echo", prompt=prompt)
        # 'answer' é obrigatório para os testes; 'output' mantido por compatibilidade
        return {"provider": "echo", "answer": prompt, "output": prompt}

    # OPENAI
    if provider == "openai":
        if not openai_configured():
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY não configurada.")
        try:
            resp = ask_openai(prompt)
            # resp já vem normalizado: {provider, model, answer, usage}
            return resp
        except RuntimeError as e:
            # erro do provedor → 502 Bad Gateway
            raise HTTPException(status_code=502, detail=str(e)) from e

    # GEMINI
    if provider == "gemini":
        if not gemini_configured():
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada.")
        try:
            resp = ask_gemini(prompt)
            # resp já vem normalizado: {provider, model, answer, usage}
            return resp
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    # Provider inválido
    raise HTTPException(status_code=400, detail=f"Provider não suportado: {provider}")
