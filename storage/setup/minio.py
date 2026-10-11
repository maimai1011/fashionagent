"""MinIO connection and bucket setup."""

import logging
import os
from urllib.parse import urlsplit

from minio import Minio


logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    logger.info("function_start function=setup.minio._required_env")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def get_client() -> Minio:
    """Create a MinIO client from environment configuration."""
    logger.info("function_start function=setup.minio.get_client")
    raw_endpoint = _required_env("MINIO_ENDPOINT")
    secure = os.getenv("MINIO_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if "://" in raw_endpoint:
        parsed_endpoint = urlsplit(raw_endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            raise ValueError("MINIO_ENDPOINT must be a valid HTTP(S) endpoint")
        endpoint = parsed_endpoint.netloc
        secure = parsed_endpoint.scheme == "https"
    else:
        endpoint = raw_endpoint.rstrip("/")

    logger.info("Connecting to MinIO at %s", endpoint)
    return Minio(
        endpoint=endpoint,
        access_key=_required_env("MINIO_ACCESS_KEY"),
        secret_key=_required_env("MINIO_SECRET_KEY"),
        secure=secure,
    )


def create_bucket(bucket_name: str) -> None:
    """Create a bucket when it does not already exist."""
    logger.info("function_start function=setup.minio.create_bucket")
    if not bucket_name or not bucket_name.strip():
        raise ValueError("bucket_name cannot be empty")

    client = get_client()
    if client.bucket_exists(bucket_name):
        logger.info("MinIO bucket '%s' already exists", bucket_name)
        return

    client.make_bucket(bucket_name)
    logger.info("Created MinIO bucket '%s'", bucket_name)


