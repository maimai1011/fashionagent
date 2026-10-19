"""Shared request dependencies for the public API."""

from typing import Annotated

from fastapi import Header, HTTPException


def request_user_id(
    x_user_id: Annotated[str, Header(min_length=1, max_length=200)],
) -> str:
    """Return the caller-supplied identity used to isolate user data."""
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="X-User-ID cannot be empty")
    return user_id
