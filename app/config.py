# app/config.py
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações centralizadas do orquestrador-ai.
    Carrega do ambiente e opcionalmente de um arquivo .env na raiz do projeto.
    """

    # App
    APP_NAME: str = Field(default="orquestrador-ai")
    APP_VERSION: str = Field(default="0.1.0")
    LOG_LEVEL: str = Field(default="INFO")  # DEBUG | INFO | WARNING | ERROR

    # Providers (Sprint 3)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")

    # Orquestração
    DEFAULT_PROVIDER: str = Field(default="openai")  # openai | gemini | echo
    PROVIDER_FALLBACK: List[str] = Field(
        default_factory=lambda: ["openai", "gemini"]
    )  # ordem de fallback

    # Timeouts (segundos)
    HTTP_TIMEOUT: float = Field(default=30.0)
    PROVIDER_TIMEOUT: float = Field(default=25.0)

    # Cache / Redis (opcional, para sprints futuras)
    REDIS_DSN: Optional[str] = Field(default=None)  # ex: redis://localhost:6379/0
    CACHE_TTL_DEFAULT: int = Field(default=60)  # segundos

    # Métricas
    METRICS_PATH: str = Field(default="/metrics")

    model_config = SettingsConfigDict(
        env_file=".env",           # carrega variáveis do .env se existir
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",            # ignora variáveis extras
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna uma instância única de Settings (cacheada).
    Use:
        from app.config import settings
    """
    return Settings()


# Instância pronta para import direto
settings = get_settings()
