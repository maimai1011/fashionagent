"""Low-cardinality application metrics exported through OpenTelemetry."""

from functools import lru_cache


@lru_cache
def _instruments():
    from opentelemetry import metrics

    meter = metrics.get_meter("fashion.observability")
    return {
        "http_count": meter.create_counter("fashion.http.requests", unit="1"),
        "http_duration": meter.create_histogram("fashion.http.duration", unit="s"),
        "chat_count": meter.create_counter("fashion.chat.turns", unit="1"),
        "chat_duration": meter.create_histogram("fashion.chat.duration", unit="s"),
        "intent_count": meter.create_counter("fashion.intent.decisions", unit="1"),
        "stylist_count": meter.create_counter("fashion.stylist.calls", unit="1"),
        "stylist_duration": meter.create_histogram("fashion.stylist.duration", unit="s"),
        "wardrobe_count": meter.create_counter("fashion.wardrobe.tasks", unit="1"),
        "wardrobe_duration": meter.create_histogram("fashion.wardrobe.duration", unit="s"),
    }


def _record(counter: str, duration: str | None, seconds: float, **labels: str) -> None:
    try:
        instruments = _instruments()
    except ModuleNotFoundError:
        return  # The agent-only package can run without the API telemetry dependencies.
    instruments[counter].add(1, labels)
    if duration:
        instruments[duration].record(seconds, labels)


def record_http(method: str, route: str, status_code: int, seconds: float) -> None:
    _record(
        "http_count", "http_duration", seconds,
        method=method, route=route, status_code=str(status_code),
    )


def record_chat(outcome: str, seconds: float) -> None:
    _record("chat_count", "chat_duration", seconds, outcome=outcome)


def record_intent(outcome: str) -> None:
    _record("intent_count", None, 0, outcome=outcome)


def record_stylist(outcome: str, seconds: float) -> None:
    _record("stylist_count", "stylist_duration", seconds, outcome=outcome)


def record_wardrobe(task: str, outcome: str, seconds: float) -> None:
    _record("wardrobe_count", "wardrobe_duration", seconds, task=task, outcome=outcome)
