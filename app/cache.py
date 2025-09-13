import redis
from typing import Optional
from app.config import settings

_redis_client: Optional[redis.Redis] = None

def get_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client:
        return _redis_client
    if not settings.redis_url:
        return None
    _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client

def cache_get(key: str) -> Optional[str]:
    client = get_client()
    if not client:
        return None
    try:
        return client.get(key)
    except Exception:
        return None

def cache_set(key: str, value: str, ttl_seconds: int = 300) -> None:
    client = get_client()
    if not client:
        return
    try:
        client.setex(key, ttl_seconds, value)
    except Exception:
        pass
