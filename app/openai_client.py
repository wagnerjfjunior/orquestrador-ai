import httpx
from app.config import settings

OPENAI_BASE = "https://api.openai.com/v1"

async def chat_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not settings.openai_api_key:
        return "[openai] missing OPENAI_API_KEY"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    timeout = httpx.Timeout(20, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
