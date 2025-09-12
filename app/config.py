from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    redis_url: str | None = None

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()
