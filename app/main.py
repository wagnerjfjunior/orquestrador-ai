from fastapi import FastAPI, Body
import os
import redis

app = FastAPI(title="Orquestrador AI", version="0.1.0")

# Rota raiz
@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

# Healthcheck
@app.get("/health")
async def health():
    return {"status": "ok"}

# Teste de cache Redis (produção no Render)
@app.get("/cache-test")
async def cache_test():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return {"redis": "missing REDIS_URL"}

    try:
        r = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=2,      # timeouts curtos p/ não travar
            socket_connect_timeout=2,
        )
        key = "orq:ping"
        r.set(key, "pong", ex=60)
        val = r.get(key)
        return {"redis": "ok", "value": val}
    except Exception as e:
        return {"redis": "error", "error": str(e)}

# Endpoint de teste para validação da pipeline
from fastapi import FastAPI, Body, Query
import os
import redis
from app.config import settings
from app.clients.openai_client import chat_openai
from app.clients.gemini_client import chat_gemini

app = FastAPI(title="Orquestrador AI", version="0.3.0")

@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/cache-test")
async def cache_test():
    redis_url = os.getenv("REDIS_URL") or settings.redis_url
    if not redis_url:
        return {"redis": "missing REDIS_URL"}
    try:
        r = redis.from_url(redis_url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
        r.set("orq:ping", "pong", ex=60)
        return {"redis": "ok", "value": r.get("orq:ping")}
    except Exception as e:
        return {"redis": "error", "error": str(e)}

@app.post("/ask")
async def ask(
    payload: dict = Body(default={}),
    provider: str = Query("openai", enum=["openai","gemini"]),
    model: str | None = Query(None),
    cache_ttl: int = Query(300),
):
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        return {"error": "missing prompt in JSON body"}

    # chave de cache
    key = f"ask:{provider}:{(model or 'default')}:{hash(prompt)}"
    redis_url = os.getenv("REDIS_URL") or settings.redis_url
    r = redis.from_url(redis_url, decode_responses=True) if redis_url else None

    # tenta cache
    if r:
        cached = r.get(key)
        if cached:
            return {"provider": provider, "model": model or "default", "cached": True, "output": cached}

    # chama provedor
    if provider == "openai":
        text = await chat_openai(prompt, model or "gpt-4o-mini")
    else:
        text = await chat_gemini(prompt, model or "gemini-1.5-flash-latest")

    # guarda no cache
    if r and text and not text.startswith("["):
        r.set(key, text, ex=cache_ttl)

    return {"provider": provider, "model": model or "default", "cached": False, "output": text}

