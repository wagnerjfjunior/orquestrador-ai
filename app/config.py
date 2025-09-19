# =============================================================================
# File: app/config.py
# Version: 2025-09-14 22:45:00 -03 (America/Sao_Paulo)
# Changes:
# - Adicionadas as variáveis de configuração para as novas estratégias de duelo.
# - Adicionado MAX_CALLS_PER_REQUEST como um guardrail de custo e performance.
# - A versão da App foi incrementada para 0.2.0.
# =============================================================================
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações centralizadas do orquestrador-ai.
    """
    # App
    APP_NAME: str = Field(default="orquestrador-ai")
    APP_VERSION: str = Field(default="0.2.0") # Versão incrementada
    LOG_LEVEL: str = Field(default="INFO")

    # Providers
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")

    # Orquestração (NOVAS CONFIGURAÇÕES DA SPRINT)
    ALLOWED_STRATEGIES: List[str] = Field(
        default_factory=lambda: ["heuristic", "crossvote", "refine_once_crossvote"]
    )
    DEFAULT_STRATEGY: str = Field(default="heuristic")
    MAX_CALLS_PER_REQUEST: int = Field(default=6, description="Limite de segurança para chamadas de IA num único pedido.")

    # Timeouts (segundos)
    PROVIDER_TIMEOUT: float = Field(default=25.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

