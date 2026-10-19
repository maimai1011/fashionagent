"""Celery tasks for asynchronous wardrobe image inference."""

import logging
import os
import time
from io import BytesIO
from urllib.parse import quote
from uuid import UUID

from celery import Celery, Task
from celery.signals import worker_process_init
from PIL import Image, ImageOps

from agent.observability.context import request_id_var, session_id_var, task_id_var
from agent.observability.logging import configure_logging
from agent.observability.metrics import record_wardrobe
from agent.observability.telemetry import instrument_celery_worker, trace_span
from inference.image_inference.image_processing import ImageProcessing
from storage.processing.wardrobe_embeddings import insert_wardrobe_embedding
from storage.processing.wardrobe_repository import (
    complete_wardrobe_processing,
    get_wardrobe_image,
    update_wardrobe_status,
)
from storage.setup.minio import get_minio_client

logger = logging.getLogger(__name__)


class ObservedTask(Task):
    """Restore HTTP correlation fields while a task executes."""

    def __call__(self, *args, **kwargs):
        headers = self.request.headers or {}
        request_token = request_id_var.set(headers.get("x-request-id", ""))
        session_token = session_id_var.set(headers.get("x-session-id", ""))
        task_token = task_id_var.set(self.request.id or "")
        try:
            return super().__call__(*args, **kwargs)
        finally:
            task_id_var.reset(task_token)
            session_id_var.reset(session_token)
            request_id_var.reset(request_token)


@worker_process_init.connect(weak=False)
def initialize_worker_observability(**_kwargs) -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"), service_name="fashion-worker")
    instrument_celery_worker()


def _broker_url() -> str:
    explicit_url = os.getenv("CELERY_BROKER_URL")
    if explicit_url:
        return explicit_url
    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = os.environ["RABBITMQ_PORT"]
    return f"amqp://guest:guest@{host}:{port}//"


celery_app = Celery(
    "fashion-wardrobe",
    broker=_broker_url(),
    backend=os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
)
celery_app.Task = ObservedTask
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="storage.embed_wardrobe_queries")
def embed_wardrobe_queries(texts: list[str]) -> list[list[float]]:
    """Use the upload worker's CLIP model for wardrobe text searches."""
    from inference.image_inference.model import get_clothing_model

    started = time.perf_counter()
    try:
        result = get_clothing_model().embed_texts(texts)
    except Exception:
        record_wardrobe("embed_queries", "failed", time.perf_counter() - started)
        raise
    record_wardrobe("embed_queries", "completed", time.perf_counter() - started)
    return result


# Keep the registered name so tasks already queued before this move still run.
@celery_app.task(name="storage.process_wardrobe_image", ignore_result=True)
def process_wardrobe_image(item_id: str) -> dict[str, str]:
    """Process one newly uploaded wardrobe image and persist its result."""
    started = time.perf_counter()
    wardrobe_item_id = UUID(item_id)

    try:
        update_wardrobe_status(wardrobe_item_id, "processing")
        item = get_wardrobe_image(wardrobe_item_id)
        bucket = _required_env("MINIO_BUCKET")
        client = get_minio_client()
        with trace_span("fashion.wardrobe.fetch_image"):
            response = client.get_object(Bucket=bucket, Key=item.raw_object_key)
            raw_image = response["Body"].read()
            response["Body"].close()

        with Image.open(BytesIO(raw_image)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

        from inference.image_inference.model import get_clothing_model

        with trace_span("fashion.wardrobe.inference"):
            processed_image, color, clothing_type, embedding = (
                ImageProcessing().run_image_pipeline(image, get_clothing_model())
            )
        output = BytesIO()
        processed_image.save(output, format="PNG")

        user_path = quote(item.user_id, safe="")
        processed_key = f"processed/{user_path}/{item.item_id}.png"
        with trace_span("fashion.wardrobe.persist"):
            client.put_object(
                Bucket=bucket,
                Key=processed_key,
                Body=output.getvalue(),
                ContentType="image/png",
            )
            embedding_ref = insert_wardrobe_embedding(wardrobe_item_id, embedding)
            complete_wardrobe_processing(
                wardrobe_item_id,
                processed_key,
                color,
                clothing_type,
                embedding_ref,
            )
        latency_seconds = time.perf_counter() - started
        logger.info(
            "wardrobe_processing_completed item_id=%s latency_ms=%.2f",
            wardrobe_item_id, latency_seconds * 1000,
        )
        record_wardrobe("process_image", "completed", latency_seconds)
        return {
            "item_id": item_id,
            "status": "completed",
            "processed_object_key": processed_key,
            "dominant_color": color,
            "clothing_type": clothing_type,
            "embedding_ref": embedding_ref,
        }
    except Exception as exc:
        logger.exception("Wardrobe image processing failed for '%s'", item_id)
        record_wardrobe("process_image", "failed", time.perf_counter() - started)
        update_wardrobe_status(wardrobe_item_id, "failed", str(exc)[:1_000])
        raise


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value
