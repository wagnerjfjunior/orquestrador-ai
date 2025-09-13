from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Metadados do app
    app_name: str = "Orquestrador AI"
    version: str = "0.1.0"

    # Chaves/API
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Redis
    redis_url: Optional[str] = None

    # Config de leitura de env (.env local e variáveis do ambiente)
    model_config = SettingsConfigDict(
        env_prefix="",          # não exige prefixo (usa nomes exatos)
        env_file=".env",        # útil em dev local
        case_sensitive=False    # permite OPENAI_API_KEY ou openai_api_key
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
