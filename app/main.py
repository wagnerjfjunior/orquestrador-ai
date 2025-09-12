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
@app.post("/ask")
async def ask(payload: dict = Body(default={})):
    return {"message": "Hello World", "echo": payload}
