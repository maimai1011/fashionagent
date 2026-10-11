"""Functions for storing raw user data."""

from typing import Any


__all__ = [
    "StoredImage",
    "UserProfile",
    "insert_image",
    "insert_user_profile",
]


def __getattr__(name: str) -> Any:
    """Load image-storage dependencies only when they are requested."""
    if name in {"StoredImage", "insert_image"}:
        from .images import StoredImage, insert_image

        return {"StoredImage": StoredImage, "insert_image": insert_image}[name]
    if name in {"UserProfile", "insert_user_profile"}:
        from .profile import UserProfile, insert_user_profile

        return {
            "UserProfile": UserProfile,
            "insert_user_profile": insert_user_profile,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
