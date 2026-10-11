"""Store a raw clothing image and register it in PostgreSQL."""

import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote
from uuid import UUID, uuid4

from raw_data.query import INSERT_USER_IMAGE
from setup.minio import get_client
from setup.postgres import get_connection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredImage:
    """Identifiers returned after an image is stored successfully."""

    id: UUID
    user_id: str
    bucket: str
    object_key: str
    original_filename: str
    content_type: str


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def insert_image(
    user_id: str,
    image: BinaryIO,
    size: int,
    original_filename: str,
    content_type: str | None = None,
) -> StoredImage:
    """Upload one clothing image to MinIO and insert its PostgreSQL record.

    The user must already exist in ``user_profiles``. If the database insert
    fails, the uploaded MinIO object is removed to avoid leaving orphan data.
    """
    clean_user_id = user_id.strip()
    clean_filename = Path(original_filename).name
    if not clean_user_id:
        raise ValueError("user_id cannot be empty")
    if not clean_filename:
        raise ValueError("original_filename cannot be empty")
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if not hasattr(image, "read"):
        raise TypeError("image must be a readable binary stream")

    detected_type = content_type or mimetypes.guess_type(clean_filename)[0]
    final_content_type = detected_type or "application/octet-stream"
    if not final_content_type.startswith("image/"):
        raise ValueError("content_type must be an image type")

    suffix = Path(clean_filename).suffix.lower()
    user_path = quote(clean_user_id, safe="")
    object_key = f"raw_data/{user_path}/{uuid4().hex}{suffix}"
    bucket = _required_env("MINIO_BUCKET")
    minio_client = get_client()

    minio_client.put_object(
        bucket_name=bucket,
        object_name=object_key,
        data=image,
        length=size,
        content_type=final_content_type,
    )

    connection = None
    try:
        connection = get_connection()
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    INSERT_USER_IMAGE,
                    (
                        clean_user_id,
                        object_key,
                        clean_filename,
                        final_content_type,
                    ),
                )
                image_id = cursor.fetchone()[0]
    except Exception:
        logger.exception(
            "Database insert failed; removing uploaded object '%s'", object_key
        )
        try:
            minio_client.remove_object(bucket, object_key)
        except Exception:
            logger.exception("Could not remove orphan object '%s'", object_key)
        raise
    finally:
        if connection is not None:
            connection.close()

    logger.info("Stored raw image '%s' for user '%s'", object_key, clean_user_id)
    return StoredImage(
        id=image_id,
        user_id=clean_user_id,
        bucket=bucket,
        object_key=object_key,
        original_filename=clean_filename,
        content_type=final_content_type,
    )
