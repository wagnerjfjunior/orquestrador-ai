from fastapi import FastAPI, Body, Request
from app.config import settings
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics

# imports existentes
import httpx
import os
import redis

# ---------- inicialização ----------
app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# middleware de tracing/log
app.add_middleware(TraceMiddleware)

# métricas /metrics
setup_metrics(app)

# ---------- redis client (opcional) ----------
_redis_client = None
if settings.REDIS_URL:
    try:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.error("redis_init_error", error=str(e))

# ---------- raiz ----------
@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}

# ---------- health ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- config-check ----------
@app.get("/config-check")
async def config_check():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "has_openai_key": bool(settings.OPENAI_API_KEY),
        "has_gemini_key": bool(settings.GEMINI_API_KEY),
        "redis_url_present": bool(settings.REDIS_URL),
    }

# ---------- cache-test ----------
@app.get("/cache-test")
async def cache_test():
    if not _redis_client:
        return {"redis": "not-configured"}
    _redis_client.set("ping", "pong", ex=60)
    val = _redis_client.get("ping")
    return {"redis": "ok", "value": val}

# ---------- providers simples ----------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

async def chat_gemini(prompt: str, model: str = "gemini-1.5-flash-latest") -> str:
    if not settings.GEMINI_API_KEY:
        return "[gemini] missing GEMINI_API_KEY"
    params = {"key": settings.GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    timeout = httpx.Timeout(20, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{GEMINI_BASE}/models/{model}:generateContent", params=params, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"[gemini] unexpected response: {data}"

async def chat_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    # chamando REST sem SDK para manter simples
    if not settings.OPENAI_API_KEY:
        return "[openai] missing OPENAI_API_KEY or SDK not available"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    timeout = httpx.Timeout(20, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            return f"[openai] http_error {r.status_code}: {r.text}"
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return f"[openai] unexpected response: {data}"

# ---------- endpoint principal ----------
@app.post("/ask")
async def ask(request: Request, payload: dict = Body(default={})):
    # trace_id para resposta e logs
    trace_id = request.headers.get("X-Trace-Id")
    prompt = (payload or {}).get("prompt", "")
    provider = (request.query_params.get("provider") or "echo").lower()

    # cache opcional e simples (apenas echo com key "echo:{prompt}")
    ttl = int(request.query_params.get("ttl") or "0")
    cache_key = None
    if provider == "echo" and _redis_client and prompt and ttl > 0:
        cache_key = f"echo:{prompt}"
        cached = _redis_client.get(cache_key)
        if cached:
            # marca no header que veio do cache (para você depurar com curl -i)
            headers = {"X-Cache": "HIT", "X-Trace-Id": trace_id or ""}
            return {"provider": provider, "answer": cached, "trace_id": trace_id}, 200, headers

    # roteamento simples
    if provider == "echo":
        answer = prompt
    elif provider == "gemini":
        answer = await chat_gemini(prompt)
    elif provider == "openai":
        answer = await chat_openai(prompt)
    else:
        answer = f"[error] unknown provider={provider}"

    # grava cache se aplicável
    headers = {"X-Cache": "MISS", "X-Trace-Id": trace_id or ""}
    if cache_key and _redis_client and ttl > 0 and answer:
        try:
            _redis_client.set(cache_key, answer, ex=ttl)
        except Exception as e:
            logger.error("redis_set_error", error=str(e))

    return {"provider": provider, "answer": answer, "trace_id": trace_id}, 200, headers
