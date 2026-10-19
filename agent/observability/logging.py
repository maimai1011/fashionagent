"""JSON logging shared by the API and image worker."""

import json
import logging
import os
from datetime import UTC, datetime

from agent.observability.context import request_id_var, session_id_var, task_id_var


class JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        trace_id = ""
        span_id = ""
        try:
            from opentelemetry import trace

            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                trace_id = format(span_context.trace_id, "032x")
                span_id = format(span_context.span_id, "016x")
        except ImportError:
            pass
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "task_id": task_id_var.get(),
            "trace_id": trace_id,
            "span_id": span_id,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, *, service_name: str = "fashion-api") -> None:
    """Write one JSON object per line to standard error."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(service_name))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def agent_content_logging_enabled() -> bool:
    """Allow explicit, temporary logging of chat text and agent payloads."""
    return os.getenv("AGENT_LOG_CONTENT", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
