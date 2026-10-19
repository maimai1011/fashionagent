"""Checks for the asynchronous wardrobe-result API."""

import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from psycopg2.extensions import adapt
from starlette.datastructures import Headers

from agent.observability.context import request_id_var
from api import wardrobe
from storage.processing import wardrobe_repository


class WardrobeResultTests(unittest.IsolatedAsyncioTestCase):
    def test_postgres_accepts_uuid_item_ids(self) -> None:
        item_id = uuid4()
        self.assertEqual(adapt(item_id).getquoted(), f"'{item_id}'::uuid".encode())

    async def test_upload_response_links_to_result(self) -> None:
        item_id = uuid4()
        stored = SimpleNamespace(
            id=item_id,
            user_id="user-123",
            object_key="raw_data/user-123/shirt.png",
            original_filename="shirt.png",
            content_type="image/png",
        )
        image = UploadFile(
            file=BytesIO(b"image-data"),
            filename="shirt.png",
            headers=Headers({"content-type": "image/png"}),
        )
        request_token = request_id_var.set("request-123")
        with (
            patch.object(wardrobe, "insert_image", return_value=stored),
            patch.object(wardrobe, "update_wardrobe_status"),
            patch.object(
                wardrobe.process_wardrobe_image,
                "apply_async",
                return_value=SimpleNamespace(id="task-1"),
            ) as enqueue,
        ):
            try:
                response = await wardrobe.upload_wardrobe_item(image, "user-123")
            finally:
                request_id_var.reset(request_token)
        self.assertEqual(response.status, "queued")
        self.assertEqual(response.result_url, f"/v1/wardrobe/{item_id}")
        enqueue.assert_called_once_with(
            args=[str(item_id)], headers={"x-request-id": "request-123"},
        )

    async def test_completed_result_exposes_image_endpoint(self) -> None:
        item_id = uuid4()
        item = wardrobe_repository.WardrobeResult(
            item_id=item_id,
            user_id="user-123",
            status="completed",
            original_filename="shirt.png",
            processed_object_key=f"processed/user-123/{item_id}.png",
            dominant_color="blue",
            clothing_type="shirt",
            error_message=None,
            raw_object_key=f"raw_data/user-123/{item_id}.png",
            content_type="image/png",
        )
        with patch.object(wardrobe, "get_wardrobe_result", return_value=item) as get_result:
            response = await wardrobe.read_wardrobe_result(item_id, "user-123")

        get_result.assert_called_once_with(item_id, "user-123")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.dominant_color, "blue")
        self.assertEqual(response.clothing_type, "shirt")
        self.assertEqual(response.image_url, f"/v1/wardrobe/{item_id}/image")
        self.assertEqual(response.processed_image_url, f"/v1/wardrobe/{item_id}/image")

        with (
            patch.object(wardrobe, "get_wardrobe_result", return_value=item),
            patch.object(wardrobe, "get_processed_image", return_value=b"png") as get_image,
        ):
            image_response = await wardrobe.read_processed_wardrobe_image(item_id, "user-123")
        get_image.assert_called_once_with(item.processed_object_key)
        self.assertEqual(image_response.body, b"png")
        self.assertEqual(image_response.media_type, "image/png")

    async def test_pending_result_has_no_image(self) -> None:
        item_id = uuid4()
        item = wardrobe_repository.WardrobeResult(
            item_id=item_id,
            user_id="user-123",
            status="processing",
            original_filename="shirt.png",
            processed_object_key=None,
            dominant_color=None,
            clothing_type=None,
            error_message=None,
            raw_object_key=f"raw_data/user-123/{item_id}.jpg",
            content_type="image/jpeg",
        )
        with patch.object(wardrobe, "get_wardrobe_result", return_value=item):
            response = await wardrobe.read_wardrobe_result(item_id, "user-123")
            with self.assertRaises(HTTPException) as raised:
                await wardrobe.read_processed_wardrobe_image(item_id, "user-123")
        self.assertIsNone(response.processed_image_url)
        self.assertEqual(response.image_url, f"/v1/wardrobe/{item_id}/raw-image")
        self.assertEqual(raised.exception.status_code, 409)

        with (
            patch.object(wardrobe, "get_wardrobe_result", return_value=item),
            patch.object(wardrobe, "get_image_object", return_value=b"jpeg") as get_image,
        ):
            raw_response = await wardrobe.read_raw_wardrobe_image(item_id, "user-123")
        get_image.assert_called_once_with(item.raw_object_key)
        self.assertEqual(raw_response.body, b"jpeg")
        self.assertEqual(raw_response.media_type, "image/jpeg")

    async def test_other_users_item_is_not_returned(self) -> None:
        item_id = uuid4()
        with (
            patch.object(wardrobe, "get_wardrobe_result", return_value=None),
            self.assertRaises(HTTPException) as raised,
        ):
            await wardrobe.read_wardrobe_result(item_id, "different-user")
        self.assertEqual(raised.exception.status_code, 404)

        with (
            patch.object(wardrobe, "get_wardrobe_result", return_value=None),
            patch.object(wardrobe, "get_image_object") as get_image,
            self.assertRaises(HTTPException) as raw_error,
        ):
            await wardrobe.read_raw_wardrobe_image(item_id, "different-user")
        self.assertEqual(raw_error.exception.status_code, 404)
        get_image.assert_not_called()

    def test_database_query_is_scoped_to_user(self) -> None:
        item_id = uuid4()
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value = None
        with patch.object(wardrobe_repository, "get_connection", return_value=connection):
            result = wardrobe_repository.get_wardrobe_result(item_id, "user-123")
        self.assertIsNone(result)
        query, params = connection.cursor.return_value.__enter__.return_value.execute.call_args.args
        self.assertIn("user_id = %s", query)
        self.assertEqual(params, (item_id, "user-123"))
        connection.close.assert_called_once()

    def test_list_query_returns_user_uploads_newest_first(self) -> None:
        item_id = uuid4()
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [
            (item_id, "user-123", "completed", "shirt.png", "processed/shirt.png", "blue", "shirt", None, "raw_data/shirt.png", "image/png")
        ]
        with patch.object(wardrobe_repository, "get_connection", return_value=connection):
            items = wardrobe_repository.list_wardrobe_results("user-123")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].item_id, item_id)
        self.assertEqual(items[0].dominant_color, "blue")
        query, params = connection.cursor.return_value.__enter__.return_value.execute.call_args.args
        self.assertIn("WHERE user_id = %s", query)
        self.assertIn("ORDER BY created_at DESC", query)
        self.assertEqual(params, ("user-123",))
        connection.close.assert_called_once()

    def test_list_endpoint_returns_processing_results_for_current_user(self) -> None:
        completed_id = uuid4()
        pending_id = uuid4()
        items = [
            wardrobe_repository.WardrobeResult(
                completed_id, "user-123", "completed", "shirt.png",
                "processed/shirt.png", "blue", "shirt", None,
                "raw_data/shirt.png", "image/png",
            ),
            wardrobe_repository.WardrobeResult(
                pending_id, "user-123", "processing", "pants.png",
                None, None, None, None, "raw_data/pants.png", "image/png",
            ),
        ]
        with (
            patch.object(wardrobe, "list_wardrobe_results", return_value=items) as list_results,
            TestClient(wardrobe.create_app()) as client,
        ):
            response = client.get("/v1/wardrobe", headers={"X-User-ID": "user-123"})
            missing_header = client.get("/v1/wardrobe")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(missing_header.status_code, 422)
        list_results.assert_called_once_with("user-123")
        self.assertEqual(response.json(), [
            {
                "item_id": str(completed_id), "status": "completed",
                "original_filename": "shirt.png", "dominant_color": "blue",
                "clothing_type": "shirt",
                "image_url": f"/v1/wardrobe/{completed_id}/image",
                "processed_image_url": f"/v1/wardrobe/{completed_id}/image",
                "error_message": None,
            },
            {
                "item_id": str(pending_id), "status": "processing",
                "original_filename": "pants.png", "dominant_color": None,
                "clothing_type": None,
                "image_url": f"/v1/wardrobe/{pending_id}/raw-image",
                "processed_image_url": None,
                "error_message": None,
            },
        ])


if __name__ == "__main__":
    unittest.main()
