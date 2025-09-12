import httpx
from app.config import settings

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
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{GEMINI_BASE}/models/{model}:generateContent", params=params, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"[gemini] unexpected response: {data}"
