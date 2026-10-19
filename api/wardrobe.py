"""User wardrobe image-upload API."""

import logging
from io import BytesIO
from typing import Annotated
from uuid import UUID

import uvicorn
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from agent.observability.context import celery_headers
from api.config import api_settings
from api.dependencies import request_user_id
from inference.image_inference.tasks import process_wardrobe_image
from storage.processing.wardrobe_images import get_image_object, get_processed_image
from storage.processing.wardrobe_repository import (
    WardrobeResult,
    get_wardrobe_result,
    list_wardrobe_results,
    update_wardrobe_status,
)
from storage.raw_data.images import insert_image

logger = logging.getLogger(__name__)


class WardrobeUploadResponse(BaseModel):
    """Metadata returned after a wardrobe image is queued."""

    item_id: UUID
    user_id: str
    raw_object_key: str
    original_filename: str
    content_type: str
    status: str
    task_id: str
    result_url: str


class WardrobeResultResponse(BaseModel):
    """Current status and inference results for an uploaded image."""

    item_id: UUID
    status: str
    original_filename: str
    dominant_color: str | None
    clothing_type: str | None
    image_url: str
    processed_image_url: str | None
    error_message: str | None


def _result_response(item: WardrobeResult) -> WardrobeResultResponse:
    processed_image_url = None
    if item.status == "completed" and item.processed_object_key:
        processed_image_url = f"/v1/wardrobe/{item.item_id}/image"
    return WardrobeResultResponse(
        item_id=item.item_id,
        status=item.status,
        original_filename=item.original_filename,
        dominant_color=item.dominant_color,
        clothing_type=item.clothing_type,
        image_url=processed_image_url or f"/v1/wardrobe/{item.item_id}/raw-image",
        processed_image_url=processed_image_url,
        error_message=item.error_message,
    )


router = APIRouter(prefix="/v1", tags=["wardrobe"])


@router.post(
    "/wardrobe",
    response_model=WardrobeUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_wardrobe_item(
    image: Annotated[UploadFile, File(description="One clothing image")],
    user_id: Annotated[str, Depends(request_user_id)],
) -> WardrobeUploadResponse:
    """Store one raw image and queue its processing task."""
    content_type = image.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file must be an image",
        )

    content = await image.read(api_settings.upload_max_bytes + 1)
    await image.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(content) > api_settings.upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {api_settings.upload_max_bytes}-byte limit",
        )

    stored = await run_in_threadpool(
        insert_image,
        user_id=user_id,
        image=BytesIO(content),
        size=len(content),
        original_filename=image.filename or "wardrobe-image",
        content_type=content_type,
    )

    try:
        await run_in_threadpool(update_wardrobe_status, stored.id, "queued")
        task = await run_in_threadpool(
            process_wardrobe_image.apply_async,
            args=[str(stored.id)],
            headers=celery_headers(),
        )
    except Exception as exc:
        await run_in_threadpool(
            update_wardrobe_status,
            stored.id,
            "queue_failed",
            str(exc)[:1_000],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image was stored, but the processing queue is unavailable",
        ) from exc

    logger.info("wardrobe_upload_queued item_id=%s task_id=%s", stored.id, task.id)

    return WardrobeUploadResponse(
        item_id=stored.id,
        user_id=stored.user_id,
        raw_object_key=stored.object_key,
        original_filename=stored.original_filename,
        content_type=stored.content_type,
        status="queued",
        task_id=task.id,
        result_url=f"/v1/wardrobe/{stored.id}",
    )


@router.get("/wardrobe", response_model=list[WardrobeResultResponse])
async def list_wardrobe_items(
    user_id: Annotated[str, Depends(request_user_id)],
) -> list[WardrobeResultResponse]:
    """List the current user's clothing and image-processing results."""
    items = await run_in_threadpool(list_wardrobe_results, user_id)
    return [_result_response(item) for item in items]


@router.get("/wardrobe/{item_id}", response_model=WardrobeResultResponse)
async def read_wardrobe_result(
    item_id: UUID,
    user_id: Annotated[str, Depends(request_user_id)],
) -> WardrobeResultResponse:
    """Return the processing result, or its current status if still pending."""
    item = await run_in_threadpool(get_wardrobe_result, item_id, user_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Wardrobe item not found")
    return _result_response(item)


@router.get("/wardrobe/{item_id}/image", response_class=Response)
async def read_processed_wardrobe_image(
    item_id: UUID,
    user_id: Annotated[str, Depends(request_user_id)],
) -> Response:
    """Serve the processed PNG for a completed item owned by the user."""
    item = await run_in_threadpool(get_wardrobe_result, item_id, user_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Wardrobe item not found")
    if item.status != "completed" or not item.processed_object_key:
        raise HTTPException(status_code=409, detail="Processed image is not ready")

    try:
        image = await run_in_threadpool(get_processed_image, item.processed_object_key)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Processed image could not be retrieved",
        ) from exc
    return Response(content=image, media_type="image/png")


@router.get("/wardrobe/{item_id}/raw-image", response_class=Response)
async def read_raw_wardrobe_image(
    item_id: UUID,
    user_id: Annotated[str, Depends(request_user_id)],
) -> Response:
    """Serve the original upload for a wardrobe item owned by the user."""
    item = await run_in_threadpool(get_wardrobe_result, item_id, user_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Wardrobe item not found")

    try:
        image = await run_in_threadpool(get_image_object, item.raw_object_key)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail="Original image could not be retrieved") from exc

    media_type = item.content_type
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif", "image/avif"}:
        media_type = "application/octet-stream"
    return Response(
        content=image,
        media_type=media_type,
        headers={"X-Content-Type-Options": "nosniff"},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Fashion Agent Wardrobe API")
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host=api_settings.host, port=api_settings.port)
