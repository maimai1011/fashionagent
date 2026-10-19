"""PostgreSQL access for wardrobe image-processing state."""

from dataclasses import dataclass
from uuid import UUID

from storage.raw_data.query import (
    COMPLETE_WARDROBE_PROCESSING,
    SELECT_WARDROBE_IMAGE,
    SELECT_WARDROBE_RESULT,
    SELECT_WARDROBE_RESULTS,
    UPDATE_WARDROBE_STATUS,
)
from storage.setup.postgres import get_connection


@dataclass(frozen=True)
class WardrobeImage:
    """A raw wardrobe image waiting to be processed."""

    item_id: UUID
    user_id: str
    raw_object_key: str
    content_type: str


@dataclass(frozen=True)
class WardrobeResult:
    """Processing state and user-facing inference results for one image."""

    item_id: UUID
    user_id: str
    status: str
    original_filename: str
    processed_object_key: str | None
    dominant_color: str | None
    clothing_type: str | None
    error_message: str | None
    raw_object_key: str
    content_type: str


def get_wardrobe_image(item_id: UUID) -> WardrobeImage:
    """Return the wardrobe image identified by ``item_id``."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_WARDROBE_IMAGE, (item_id,))
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(f"wardrobe item '{item_id}' does not exist")
    return WardrobeImage(*row)


def get_wardrobe_result(item_id: UUID, user_id: str) -> WardrobeResult | None:
    """Return an item only when it belongs to the requested user."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_WARDROBE_RESULT, (item_id, user_id))
            row = cursor.fetchone()
    finally:
        connection.close()

    return WardrobeResult(*row) if row is not None else None


def list_wardrobe_results(user_id: str) -> list[WardrobeResult]:
    """Return all of a user's uploads, newest first, with processing results."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_WARDROBE_RESULTS, (user_id,))
            rows = cursor.fetchall()
    finally:
        connection.close()

    return [WardrobeResult(*row) for row in rows]


def update_wardrobe_status(
    item_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update the processing status for one wardrobe image."""
    connection = get_connection()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                UPDATE_WARDROBE_STATUS,
                (status, error_message, item_id),
            )
    finally:
        connection.close()


def complete_wardrobe_processing(
    item_id: UUID,
    processed_object_key: str,
    dominant_color: str,
    clothing_type: str,
    embedding_ref: str,
) -> None:
    """Save the processed object and inference results in one database update."""
    connection = get_connection()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                COMPLETE_WARDROBE_PROCESSING,
                (
                    processed_object_key,
                    dominant_color,
                    clothing_type,
                    embedding_ref,
                    item_id,
                ),
            )
    finally:
        connection.close()
