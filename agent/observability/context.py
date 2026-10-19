"""Correlation fields propagated between HTTP requests and Celery tasks."""

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
task_id_var: ContextVar[str] = ContextVar("task_id", default="")


def celery_headers() -> dict[str, str]:
    """Copy nonempty correlation fields into Celery message headers."""
    return {
        key: value
        for key, value in (
            ("x-request-id", request_id_var.get()),
            ("x-session-id", session_id_var.get()),
        )
        if value
    }
