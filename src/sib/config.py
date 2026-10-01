"""Settings come from environment variables only (never hard-code keys)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    vllm_url: str = ""
    stripe_key: str = ""  # sk_test_... (test mode); live keys also need STRIPE_ALLOW_LIVE=true
    stripe_webhook_secret: str = ""
    stripe_finalize: bool = False  # leave invoices as drafts for review unless set
    stripe_allow_live: bool = False
    billing_currency: str = "usd"
    database_url: str = ""
    margin_target: float = 0.75
    live_models: bool = False  # LIVE_MODELS=true: API-tier models call the real Claude API
    state_dir: str = ".sib_state"
    ledger_dir: str = "ledger"
    evals_dir: str = "evals"


@lru_cache
def get_settings() -> Settings:
    return Settings()
