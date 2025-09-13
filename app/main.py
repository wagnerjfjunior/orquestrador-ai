from __future__ import annotations

import os
import json
import uuid
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, Body, Query, Response

from app.config import settings
from app.cache import cache_get, cache_set

app = FastAPI(title="Orquestrador AI", version="0.1.0")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ---------- Utilidades ----------
def make_trace_id() -> str:
    return uuid.uuid4().hex[:12]

def cache_key(provider: str, prompt: str, model: Optional[str]) -> str:
    m = model or ""
    return f"ask:{provider}:{m}:{hash(prompt)}"

async def chat_gemini(prompt: str, model: str = "gemini-1.5-flash-latest") -> str:
    if not settings.gemini_api_key:
        return "[gemini] missing GEMINI_API_KEY"
    params = {"key": settings.gemini_api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}
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
    api_key = settings.openai_api_key
    if not api_key:
        return "[openai] missing OPENAI_API_KEY or SDK not available"
    # uso via REST para evitar SDKs extras
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    timeout = httpx.Timeout(20, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return f"[openai] unexpected response: {data}"

# ---------- Rotas básicas ----------
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
    cache_set("ping", "pong", 60)
    val = cache_get("ping")
    return {"redis": "ok" if val == "pong" else "miss", "value": val}

# ---------- /ask com cache + trace ----------
@app.post("/ask")
async def ask(
    response: Response,
    payload: dict = Body(default={}),
    provider: Literal["echo", "gemini", "openai"] = Query(default="echo"),
    model: Optional[str] = Query(default=None),
    ttl: int = Query(default=120, ge=0, le=86400),
):
    trace_id = make_trace_id()
    prompt = (payload or {}).get("prompt") or ""
    response.headers["X-Trace-Id"] = trace_id

    # cache
    key = cache_key(provider, prompt, model)
    cached = cache_get(key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        return {"provider": provider, "answer": cached, "trace_id": trace_id}

    # roteamento simples
    if provider == "echo":
        answer = prompt
    elif provider == "gemini":
        answer = await chat_gemini(prompt, model or "gemini-1.5-flash-latest")
    elif provider == "openai":
        answer = await chat_openai(prompt, model or "gpt-4o-mini")
    else:
        answer = "[error] unknown provider"

    # salva cache (somente se houver resposta)
    if ttl and answer and not answer.startswith("[error]"):
        cache_set(key, answer, ttl)

    response.headers["X-Cache"] = "MISS"
    return {"provider": provider, "answer": answer, "trace_id": trace_id}
