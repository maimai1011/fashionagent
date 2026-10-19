"""Environment-backed chat application configuration."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _bool(name: str, default: bool = False) -> bool:
    logger.info("function_start function=_bool")
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _service_url(
    url_name: str,
    host_name: str,
    port_name: str,
    path: str = "",
) -> str:
    explicit_url = os.getenv(url_name)
    if explicit_url:
        return explicit_url
    host = os.getenv(host_name, "127.0.0.1")
    port = _port(port_name)
    return f"http://{host}:{port}{path}"


def _port(name: str) -> int:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required; set it in storage/.env")
    port = int(value)
    if not 1 <= port <= 65_535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


@dataclass(frozen=True)
class Settings:
    vllm_url: str = _service_url(
        "VLLM_URL", "VLLM_HOST", "VLLM_PORT", "/v1"
    )
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
    api_port: int = _port("API_PORT")
    api_base_url: str = _service_url(
        "API_BASE_URL", "API_HOST", "API_PORT"
    )
    api_rate_limit_per_minute: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))
    ui_host: str = os.getenv("UI_HOST", "127.0.0.1")
    ui_port: int = _port("UI_PORT")
    ui_auth_user: str | None = os.getenv("UI_AUTH_USER")
    ui_auth_password: str | None = os.getenv("UI_AUTH_PASSWORD")
    enable_ngrok: bool = _bool("ENABLE_NGROK")
    ngrok_auth_token: str | None = os.getenv("NGROK_AUTHTOKEN")


settings = Settings()


def milvus_port() -> str:
    """Return the catalog Milvus port, including late environment overrides."""
    return str(_port("MILVUS_PORT"))
