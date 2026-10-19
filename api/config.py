"""Environment-backed configuration for API servers."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    """Runtime settings shared by standalone API entrypoints."""

    host: str
    port: int
    upload_max_bytes: int

    @classmethod
    def from_env(cls) -> "ApiSettings":
        host = os.getenv("API_HOST", "127.0.0.1").strip()
        if not host:
            raise ValueError("API_HOST cannot be empty")

        raw_port = os.getenv("API_PORT")
        if not raw_port:
            raise ValueError("API_PORT is required; set it in storage/.env")
        port = int(raw_port)
        if not 1 <= port <= 65_535:
            raise ValueError("API_PORT must be between 1 and 65535")

        upload_max_bytes = int(os.getenv("UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))
        if upload_max_bytes <= 0:
            raise ValueError("UPLOAD_MAX_BYTES must be greater than zero")

        return cls(
            host=host,
            port=port,
            upload_max_bytes=upload_max_bytes,
        )


api_settings = ApiSettings.from_env()
