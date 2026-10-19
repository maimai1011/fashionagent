"""Read processed wardrobe images from MinIO."""

import os

from storage.setup.minio import get_minio_client


def get_image_object(object_key: str) -> bytes:
    """Return an image object, closing the MinIO response body."""
    bucket = os.getenv("MINIO_BUCKET")
    if not bucket:
        raise RuntimeError("MINIO_BUCKET environment variable is not set")

    response = get_minio_client().get_object(Bucket=bucket, Key=object_key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def get_processed_image(object_key: str) -> bytes:
    """Return the processed PNG for a wardrobe item."""
    return get_image_object(object_key)
