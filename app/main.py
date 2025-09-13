import httpx
import redis
import hashlib
import json
import time
from fastapi import FastAPI, Body, Query, Request
from app.config import settings

# Inicializa FastAPI
app = FastAPI(title="Orquestrador AI", version="0.1.0")

# Redis client
r = redis.from_url(settings.redis_url, decode_responses=True)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- Providers ---
async def chat_gemini(prompt: str, model: str = "gemini-1.5-flash-latest") -> str:
    if not settings.gemini_api_key:
        return "[gemini] missing GEMINI_API_KEY"
    params = {"key": settings.gemini_api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    timeout = httpx.Timeout(20, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r_ = await client.post(
            f"{GEMINI_BASE}/models/{model}:generateContent",
            params=params,
            json=payload,
        )
        r_.raise_for_status()
        data = r_.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"[gemini] unexpected response: {data}"

async def chat_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        import openai
    except ImportError:
        return "[openai] SDK not available"

    if not settings.openai_api_key:
        return "[openai] missing OPENAI_API_KEY"

    client = openai.OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content

# --- Endpoints ---
@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/config-check")
async def config_check():
    return {
        "app": "Orquestrador AI",
        "version": "0.1.0",
        "has_openai_key": bool(settings.openai_api_key),
        "has_gemini_key": bool(settings.gemini_api_key),
        "redis_url_present": bool(settings.redis_url),
    }

@app.get("/cache-test")
async def cache_test():
    r.set("ping", "pong")
    return {"redis": "ok", "value": r.get("ping")}

@app.post("/ask")
async def ask(
    request: Request,
    payload: dict = Body(default={}),
    provider: str = Query(default="echo"),
    ttl: int = Query(default=0),
):
    prompt = payload.get("prompt", "")
    cache_key = hashlib.sha256(f"{provider}:{prompt}".encode()).hexdigest()

    # Se TTL > 0, tenta pegar do cache
    if ttl > 0:
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            ttl_remaining = r.ttl(cache_key)
            return {
                "provider": provider,
                "answer": data["answer"],
                "trace_id": data["trace_id"],
                "ttl_remaining": ttl_remaining,
                "x-cache": "HIT",
            }

    # Executa provider
    if provider == "echo":
        answer = prompt
    elif provider == "gemini":
        answer = await chat_gemini(prompt)
    elif provider == "openai":
        answer = await chat_openai(prompt)
    else:
        answer = f"[error] unknown provider: {provider}"

    trace_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]

    # Grava cache se TTL > 0
    if ttl > 0:
        r.setex(cache_key, ttl, json.dumps({"answer": answer, "trace_id": trace_id}))

    return {
        "provider": provider,
        "answer": answer,
        "trace_id": trace_id,
        "ttl_remaining": ttl if ttl > 0 else None,
        "x-cache": "MISS",
    }
