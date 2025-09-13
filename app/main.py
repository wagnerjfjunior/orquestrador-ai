from fastapi import FastAPI, Body, Query
import httpx
from app.config import settings

# -----------------------------
# Helpers LLM (mínimos)
# -----------------------------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

async def chat_gemini(prompt: str, model: str = "gemini-1.5-flash-latest") -> str:
    if not settings.gemini_api_key:
        return "[gemini] missing GEMINI_API_KEY"
    params = {"key": settings.gemini_api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    timeout = httpx.Timeout(20, connect=5)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{GEMINI_BASE}/models/{model}:generateContent",
                params=params,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[gemini] error: {e!r}"


async def chat_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
    except Exception:
        return "[openai] missing SDK"

    if not settings.openai_api_key:
        return "[openai] missing OPENAI_API_KEY"

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"[openai] error: {e!r}"


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title=settings.app_name, version=settings.version)


@app.get("/")
async def root():
    return {"service": "orquestrador-ai", "status": "live"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config-check")
async def config_check():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "has_openai_key": bool(settings.openai_api_key),
        "has_gemini_key": bool(settings.gemini_api_key),
        "redis_url_present": bool(settings.redis_url),
    }


@app.post("/ask")
async def ask(
    payload: dict = Body(default={}),
    provider: str = Query(default="echo", pattern="^(echo|openai|gemini)$"),
):
    prompt = (payload or {}).get("prompt", "")

    if provider == "echo":
        return {"provider": "echo", "answer": prompt or "(vazio)"}

    if provider == "gemini":
        answer = await chat_gemini(prompt or "Hello from Gemini")
        return {"provider": "gemini", "answer": answer}

    if provider == "openai":
        answer = await chat_openai(prompt or "Hello from OpenAI")
        return {"provider": "openai", "answer": answer}

    return {"error": f"provider inválido: {provider}"}
