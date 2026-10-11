"""Run all one-time storage initialization tasks."""

import logging
import os

from .milvus import connect_milvus, create_collection, disconnect_milvus
from .minio import create_bucket
from .postgres import initialize_postgres
from .query import POSTGRES_STATEMENTS


def _required_env(name: str) -> str:
    logging.info("function_start function=setup.__main__._required_env")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def main() -> None:
    """Initialize Postgres, Milvus, and MinIO."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.info("function_start function=setup.__main__.main")
    bucket_name = _required_env("MINIO_BUCKET")

    logging.info("Initializing Postgres")
    initialize_postgres(POSTGRES_STATEMENTS)

    logging.info("Initializing Milvus")
    connect_milvus()
    try:
        create_collection()
    finally:
        disconnect_milvus()

    logging.info("Initializing MinIO")
    create_bucket(bucket_name)
    logging.info("Storage initialization completed successfully")


if __name__ == "__main__":
    main()
