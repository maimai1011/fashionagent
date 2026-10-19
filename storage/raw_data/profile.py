"""Create a user's one-time registration profile."""

import logging
from dataclasses import dataclass
from datetime import datetime

from storage.raw_data.query import INSERT_USER_PROFILE
from storage.setup.postgres import get_connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserProfile:
    """A newly registered user profile."""

    id: int
    user_id: str
    email: str
    age: int
    gender: str
    created_at: datetime


def insert_user_profile(
    user_id: str,
    email: str,
    age: int,
    gender: str,
) -> UserProfile:
    """Insert a profile once and reject an already registered ``user_id``."""
    logger.info("function_start function=insert_user_profile")
    clean_user_id = user_id.strip()
    clean_email = email.strip().lower()
    clean_gender = gender.strip().lower()

    if not clean_user_id:
        raise ValueError("user_id cannot be empty")
    if not clean_email:
        raise ValueError("email cannot be empty")
    if not 0 <= age <= 130:
        raise ValueError("age must be between 0 and 130")
    if clean_gender not in {"male", "female"}:
        raise ValueError("gender must be 'male' or 'female'")

    connection = get_connection()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                INSERT_USER_PROFILE,
                (
                    clean_user_id,
                    clean_email,
                    age,
                    clean_gender,
                ),
            )
            row = cursor.fetchone()

        if row is None:
            raise ValueError(f"user profile '{clean_user_id}' already exists")

        profile_id, created_at = row
        return UserProfile(
            id=profile_id,
            user_id=clean_user_id,
            email=clean_email,
            age=age,
            gender=clean_gender,
            created_at=created_at,
        )
    finally:
        connection.close()
