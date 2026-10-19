"""Focused checks for the wardrobe inference task and Milvus persistence."""

import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from PIL import Image

from agent.observability.context import request_id_var, session_id_var, task_id_var
from inference.image_inference import tasks
from storage.processing import wardrobe_embeddings
from storage.setup import milvus


class WardrobeProcessingTests(unittest.TestCase):
    def test_worker_restores_and_resets_correlation_context(self) -> None:
        class ProbeTask(tasks.ObservedTask):
            name = "probe-task"

            def run(self):
                return request_id_var.get(), session_id_var.get(), task_id_var.get()

        task = ProbeTask()
        task.bind(tasks.celery_app)
        task.push_request(
            id="task-789",
            headers={"x-request-id": "request-123", "x-session-id": "session-456"},
        )
        try:
            self.assertEqual(task(), ("request-123", "session-456", "task-789"))
        finally:
            task.pop_request()
        self.assertEqual(request_id_var.get(), "")
        self.assertEqual(session_id_var.get(), "")
        self.assertEqual(task_id_var.get(), "")

    def test_task_runs_inference_and_persists_results(self) -> None:
        item_id = uuid4()
        raw = BytesIO()
        Image.new("RGB", (12, 8), "blue").save(raw, format="PNG")
        body = BytesIO(raw.getvalue())
        client = MagicMock()
        client.get_object.return_value = {"Body": body}
        item = SimpleNamespace(
            item_id=item_id,
            user_id="user-001",
            raw_object_key="raw_data/user-001/original.png",
        )
        processed_image = Image.new("RGBA", (6, 4), (0, 0, 255, 128))
        fake_model = SimpleNamespace(get_clothing_model=lambda: object())

        with (
            patch.dict("sys.modules", {"inference.image_inference.model": fake_model}),
            patch.dict("os.environ", {"MINIO_BUCKET": "fashion-products"}),
            patch.object(tasks, "get_wardrobe_image", return_value=item),
            patch.object(tasks, "get_minio_client", return_value=client),
            patch.object(tasks, "update_wardrobe_status") as update_status,
            patch.object(tasks, "complete_wardrobe_processing") as complete,
            patch.object(tasks, "insert_wardrobe_embedding", return_value="image_features:42")
            as insert_embedding,
            patch.object(
                tasks.ImageProcessing,
                "run_image_pipeline",
                return_value=(processed_image, "blue", "shirt", [0.1] * 512),
            ) as pipeline,
        ):
            result = tasks.process_wardrobe_image(str(item_id))

        update_status.assert_called_once_with(item_id, "processing")
        pipeline.assert_called_once()
        insert_embedding.assert_called_once_with(item_id, [0.1] * 512)
        complete.assert_called_once_with(
            item_id,
            f"processed/user-001/{item_id}.png",
            "blue",
            "shirt",
            "image_features:42",
        )
        upload = client.put_object.call_args.kwargs
        self.assertEqual(upload["ContentType"], "image/png")
        self.assertEqual(Image.open(BytesIO(upload["Body"])).mode, "RGBA")
        self.assertEqual(result["status"], "completed")

    def test_embedding_dimension_matches_milvus_collection(self) -> None:
        collection = MagicMock()
        collection.schema.fields = [SimpleNamespace(name="embedding", params={"dim": 512})]
        collection.insert.return_value.primary_keys = [42]
        with (
            patch.object(wardrobe_embeddings, "connect_milvus"),
            patch.object(wardrobe_embeddings, "create_collection", return_value=collection),
            patch.object(wardrobe_embeddings, "disconnect_milvus") as disconnect,
        ):
            item_id = uuid4()
            reference = wardrobe_embeddings.insert_wardrobe_embedding(item_id, [0.1] * 512)

        self.assertEqual(reference, "image_features:42")
        self.assertEqual(
            collection.insert.call_args.args[0],
            [{"image_name": str(item_id), "embedding": [0.1] * 512}],
        )
        disconnect.assert_called_once()

    def test_image_embeddings_get_cosine_index(self) -> None:
        collection = MagicMock()
        collection.name = "image_features"
        collection.indexes = []

        milvus._ensure_embedding_index(collection)

        collection.create_index.assert_called_once_with(
            field_name="embedding",
            index_params={"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
        )

    def test_existing_image_index_keeps_matching_metric(self) -> None:
        collection = MagicMock()
        collection.indexes = [
            SimpleNamespace(field_name="embedding", params={"metric_type": "COSINE"})
        ]

        milvus._ensure_embedding_index(collection)

        collection.create_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
