"""Create a user's one-time registration profile."""

import logging
from dataclasses import dataclass
from datetime import datetime

from raw_data.query import INSERT_USER_PROFILE
from setup.postgres import get_connection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserProfile:
    """A newly registered user profile."""

    id: int
    user_id: str
    name: str | None
    email: str | None
    age: int | None
    preferences: str | None
    created_at: datetime


def insert_user_profile(
    user_id: str,
    name: str | None = None,
    email: str | None = None,
    age: int | None = None,
    preferences: str | None = None,
) -> UserProfile:
    """Insert a profile once and reject an already registered ``user_id``."""
    logger.info("function_start function=insert_user_profile")
    clean_user_id = user_id.strip()
    clean_name = name.strip() if name else None
    clean_email = email.strip().lower() if email else None

    if not clean_user_id:
        raise ValueError("user_id cannot be empty")
    if age is not None and age < 0:
        raise ValueError("age cannot be negative")

    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    INSERT_USER_PROFILE,
                    (
                        clean_user_id,
                        clean_name,
                        clean_email,
                        age,
                        preferences,
                    ),
                )
                row = cursor.fetchone()

        if row is None:
            raise ValueError(f"user profile '{clean_user_id}' already exists")

        profile_id, created_at = row
        return UserProfile(
            id=profile_id,
            user_id=clean_user_id,
            name=clean_name,
            email=clean_email,
            age=age,
            preferences=preferences,
            created_at=created_at,
        )
    finally:
        connection.close()
