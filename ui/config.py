"""Environment-backed UI connection settings."""

import os


def default_api_url() -> str:
    explicit = os.getenv("API_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.getenv("API_HOST", "127.0.0.1")
    port = os.getenv("API_PORT")
    if not port:
        raise ValueError("API_PORT is required; set it in storage/.env")
    return f"http://{host}:{port}"
