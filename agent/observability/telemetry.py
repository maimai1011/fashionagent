"""Optional OTLP export and framework instrumentation."""

import logging
import os
import socket
from contextlib import contextmanager

logger = logging.getLogger(__name__)
_configured_pid: int | None = None


@contextmanager
def trace_span(name: str):
    """Add a child span when OpenTelemetry is installed and configured."""
    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return
    with trace.get_tracer("fashion.application").start_as_current_span(name):
        yield


def configure_telemetry(service_name: str) -> bool:
    """Export spans and metrics when an OTLP collector endpoint is configured."""
    global _configured_pid
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        return False
    if _configured_pid == os.getpid():
        return True
    timeout = float(os.getenv("OTEL_EXPORTER_TIMEOUT_SECONDS", "2"))

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "service.instance.id": f"{socket.gethostname()}:{os.getpid()}",
    })
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=timeout),
    ))
    trace.set_tracer_provider(tracer_provider)
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", timeout=timeout),
        export_interval_millis=10_000,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
    _configured_pid = os.getpid()
    logger.info("telemetry_configured service=%s", service_name)
    return True


def instrument_fastapi(app) -> None:
    if not configure_telemetry("fashion-api"):
        return
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health")
    CeleryInstrumentor().instrument()


def instrument_celery_worker() -> None:
    if not configure_telemetry("fashion-worker"):
        return
    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument()
