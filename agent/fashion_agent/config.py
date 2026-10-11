"""Environment-backed application configuration."""

import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    vllm_url: str = os.getenv("VLLM_URL", "http://localhost:8002/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "fashion-llm")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    recursion_limit: int = int(os.getenv("AGENT_RECURSION_LIMIT", "20"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
    checkpoint_backend: str = os.getenv("CHECKPOINT_BACKEND", "memory")
    checkpoint_database_url: str | None = os.getenv("CHECKPOINT_DATABASE_URL")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8080"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
    api_rate_limit_per_minute: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))
    ui_auth_user: str | None = os.getenv("UI_AUTH_USER")
    ui_auth_password: str | None = os.getenv("UI_AUTH_PASSWORD")
    enable_ngrok: bool = _bool("ENABLE_NGROK")
    ngrok_auth_token: str | None = os.getenv("NGROK_AUTHTOKEN")


settings = Settings()
