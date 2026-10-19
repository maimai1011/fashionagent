import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_minio_endpoint() -> str:
    """Build the MinIO endpoint from environment configuration."""
    explicit_endpoint = os.getenv("MINIO_ENDPOINT")
    if explicit_endpoint:
        return explicit_endpoint

    scheme = os.getenv("MINIO_SCHEME", "http")
    host = os.getenv("MINIO_HOST", "localhost")
    port = os.environ["MINIO_PORT"]
    return f"{scheme}://{host}:{port}"


def get_minio_client(endpoint_url: str | None = None):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or get_minio_endpoint(),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def create_bucket(bucket_name: str) -> None:
    if not bucket_name or not bucket_name.strip():
        raise ValueError("bucket_name cannot be empty")

    client = get_minio_client()

    try:
        client.head_bucket(Bucket=bucket_name)
        logger.info("MinIO bucket '%s' already exists", bucket_name)
        return

    except ClientError as exc:
        status = exc.response.get(
            "ResponseMetadata", {}
        ).get("HTTPStatusCode")

        if status != 404:
            raise

    client.create_bucket(Bucket=bucket_name)
    logger.info("Created MinIO bucket '%s'", bucket_name)
