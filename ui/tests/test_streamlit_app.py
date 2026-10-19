"""Tests for the Streamlit chat API client."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui import streamlit_app
from ui.streamlit_app import (
    create_profile,
    get_processed_image,
    get_wardrobe_result,
    send_chat,
    upload_wardrobe_image,
)


class SendChatTests(unittest.TestCase):
    def test_search_result_fetches_owned_processed_image(self) -> None:
        item_id = "8f97a100-b902-4a6e-a9f2-35948f789db8"
        matches = {"top-1": {
            "requested_item": {"item_name": "shirt"},
            "products": [{
                "product_id": item_id,
                "name": "shirt.png",
                "category": "shirt",
                "color": "blue",
                "image_url": f"http://localhost:8002/v1/wardrobe/{item_id}/image",
            }],
        }}
        with (
            patch.object(streamlit_app, "get_processed_image", return_value=b"png") as fetch,
            patch.object(streamlit_app.st, "image") as show_image,
            patch.object(streamlit_app.st, "markdown") as show_text,
        ):
            streamlit_app._show_products(matches, "http://localhost:8002", "user-123")
        fetch.assert_called_once_with("http://localhost:8002", "user-123", item_id)
        show_image.assert_called_once_with(b"png", width=180)
        rendered = " ".join(call.args[0] for call in show_text.call_args_list)
        self.assertIn("shirt", rendered)
        self.assertIn("blue", rendered)
        self.assertNotIn("shirt.png", rendered)

    def test_script_imports_when_streamlit_uses_ui_directory(self) -> None:
        ui_directory = Path(__file__).resolve().parents[1]
        subprocess.run(
            [sys.executable, "-c", "import streamlit_app"],
            cwd=ui_directory,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_creates_profile_with_form_fields(self) -> None:
        with patch("ui.streamlit_app.httpx.Client") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.post.return_value.json.return_value = {"user_id": "user-123"}

            result = create_profile(
                "http://localhost:8002/", "user-123", "user@example.com", 25, "female"
            )

        self.assertEqual(result["user_id"], "user-123")
        client.post.assert_called_once_with(
            "http://localhost:8002/v1/user-profile",
            data={
                "user_id": "user-123",
                "email": "user@example.com",
                "age": "25",
                "gender": "female",
            },
        )

    def test_uploads_image_as_multipart_with_user_header(self) -> None:
        with patch("ui.streamlit_app.httpx.Client") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.post.return_value.json.return_value = {"item_id": "item-1", "status": "queued"}

            result = upload_wardrobe_image(
                "http://localhost:8002", "user-123", "coat.png", b"png", "image/png"
            )

        self.assertEqual(result["status"], "queued")
        client.post.assert_called_once_with(
            "http://localhost:8002/v1/wardrobe",
            headers={"X-User-ID": "user-123"},
            files={"image": ("coat.png", b"png", "image/png")},
        )

    def test_reads_wardrobe_result_and_image_with_user_header(self) -> None:
        item_id = "8f97a100-b902-4a6e-a9f2-35948f789db8"
        with patch("ui.streamlit_app.httpx.Client") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.get.return_value.json.return_value = {"status": "completed"}
            client.get.return_value.content = b"processed-png"

            result = get_wardrobe_result("http://localhost:8002", "user-123", item_id)
            image = get_processed_image("http://localhost:8002", "user-123", item_id)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(image, b"processed-png")
        self.assertEqual(
            client.get.call_args_list[0].args[0],
            f"http://localhost:8002/v1/wardrobe/{item_id}",
        )
        self.assertEqual(
            client.get.call_args_list[1].args[0],
            f"http://localhost:8002/v1/wardrobe/{item_id}/image",
        )
        for call in client.get.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"X-User-ID": "user-123"})

    def test_completed_item_loads_processed_image_once(self) -> None:
        item_id = "8f97a100-b902-4a6e-a9f2-35948f789db8"
        items = {item_id: {"status": "processing", "original_filename": "coat.png"}}
        result = {
            "item_id": item_id,
            "status": "completed",
            "clothing_type": "coat",
            "dominant_color": "navy",
            "processed_image_url": f"/v1/wardrobe/{item_id}/image",
        }
        with (
            patch.object(streamlit_app.st, "session_state", SimpleNamespace(wardrobe_items=items)),
            patch.object(streamlit_app, "get_wardrobe_result", return_value=result),
            patch.object(streamlit_app, "get_processed_image", return_value=b"png") as image,
        ):
            streamlit_app._refresh_wardrobe_item("http://localhost:8002", "user-123", item_id)
            streamlit_app._refresh_wardrobe_item("http://localhost:8002", "user-123", item_id)

        self.assertEqual(items[item_id]["status"], "completed")
        self.assertEqual(items[item_id]["processed_image"], b"png")
        image.assert_called_once_with("http://localhost:8002", "user-123", item_id)

    def test_sends_user_input_and_resume_to_chat_api(self) -> None:
        with patch("ui.streamlit_app.httpx.Client") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.post.return_value.json.return_value = {"message": "Hello"}

            result = send_chat(
                "http://127.0.0.1:8002/",
                "user-123",
                "thread-1",
                "I like navy",
                resume=True,
            )

        self.assertEqual(result, {"message": "Hello"})
        client.post.assert_called_once_with(
            "http://127.0.0.1:8002/v1/chat",
            headers={"X-User-ID": "user-123"},
            json={
                "thread_id": "thread-1",
                "user_input": "I like navy",
                "resume": True,
            },
        )
        client.post.return_value.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
