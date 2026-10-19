"""Correlation context and safe telemetry labels."""

import json
import logging

from agent.observability.context import (
    celery_headers,
    request_id_var,
    session_id_var,
    task_id_var,
)
from agent.observability.logging import JsonFormatter


def test_celery_headers_include_active_request_and_session():
    request_token = request_id_var.set("request-123")
    session_token = session_id_var.set("session-456")
    try:
        assert celery_headers() == {
            "x-request-id": "request-123",
            "x-session-id": "session-456",
        }
    finally:
        session_id_var.reset(session_token)
        request_id_var.reset(request_token)


def test_json_logs_include_correlation_fields():
    request_token = request_id_var.set("request-123")
    task_token = task_id_var.set("task-789")
    try:
        record = logging.LogRecord(
            name="probe", level=logging.INFO, pathname=__file__, lineno=1,
            msg="completed", args=(), exc_info=None,
        )
        payload = json.loads(JsonFormatter("fashion-worker").format(record))
    finally:
        task_id_var.reset(task_token)
        request_id_var.reset(request_token)
    assert payload["service"] == "fashion-worker"
    assert payload["request_id"] == "request-123"
    assert payload["task_id"] == "task-789"
    assert payload["message"] == "completed"
