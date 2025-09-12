from fastapi import FastAPI, Body, Query
import httpx
from app.config import settings

# Inicializa a aplicação FastAPI
app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# Rota raiz (para acessar diretamente o domínio)
@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

# Healthcheck simples
@app.get("/health")
async def health():
    return {"status": "ok"}

# Nova rota para verificar variáveis de ambiente
@app.get("/config-check")
async def config_check():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "has_openai_key": bool(settings.OPENAI_API_KEY),
        "has_gemini_key": bool(settings.GEMINI_API_KEY),
        "redis_url_present": bool(settings.REDIS_URL),
    }

# Endpoint de teste para validação da pipeline
@app.post("/ask")
async def ask(
    payload: dict = Body(default={}),
    provider: str = Query(default="echo", description="openai | gemini | echo")
):
    prompt = payload.get("prompt", "")

    # Caso padrão (ecoar a mensagem)
    if provider == "echo":
        return {"provider": "echo", "message": "Hello World", "echo": payload}

    # Resposta simulada para OpenAI (placeholder)
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            return {"provider": "openai", "answer": "[openai] missing OPENAI_API_KEY or SDK not available"}
        return {"provider": "openai", "answer": f"(simulação OpenAI) prompt='{prompt}'"}

    # Resposta simulada para Gemini (placeholder)
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            return {"provider": "gemini", "answer": "[gemini] missing GEMINI_API_KEY"}
        # Exemplo simplificado (sem chamada real ainda)
        return {"provider": "gemini", "answer": f"(simulação Gemini) prompt='{prompt}'"}

    return {"error": f"provider '{provider}' não suportado"}
