"""Settings come from environment variables only (never hard-code keys)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    vllm_url: str = ""
    stripe_key: str = ""
    database_url: str = ""
    margin_target: float = 0.75
    state_dir: str = ".sib_state"


@lru_cache
def get_settings() -> Settings:
    return Settings()
