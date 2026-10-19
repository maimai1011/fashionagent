"""Chat search embeds text without dispatching an image-processing task."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.search.wardrobe import ClipTextEmbedding


@pytest.mark.asyncio
async def test_clip_text_embedding_runs_locally_and_reuses_model(monkeypatch):
    monkeypatch.setenv("CELERY_BROKER_URL", "amqp://unreachable.invalid:5672//")
    loads = []
    calls = []

    class FakeTensor:
        def __init__(self, rows):
            self.rows = rows

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class FakeInputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id):
            loads.append(("model", model_id))
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def get_text_features(self, **inputs):
            calls.append(inputs["texts"])
            return FakeTensor([[0.0] * 511 + [1.0] for _ in inputs["texts"]])

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            loads.append(("processor", model_id))
            return cls()

        def __call__(self, *, text, return_tensors, padding, truncation):
            assert (return_tensors, padding, truncation) == ("pt", True, True)
            return FakeInputs(texts=text)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        no_grad=nullcontext,
        nn=SimpleNamespace(functional=SimpleNamespace(normalize=lambda tensor, dim: tensor)),
    )
    fake_transformers = SimpleNamespace(CLIPModel=FakeModel, CLIPProcessor=FakeProcessor)
    monkeypatch.setattr(ClipTextEmbedding, "_model", None)
    monkeypatch.setattr(ClipTextEmbedding, "_processor", None)
    monkeypatch.setattr(ClipTextEmbedding, "_device", None)

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
        embedding = ClipTextEmbedding()
        assert await embedding.embed(["blue shirt", "black shoes"]) == [
            [0.0] * 511 + [1.0],
            [0.0] * 511 + [1.0],
        ]
        assert len(await embedding.embed(["white skirt"])) == 1

    assert loads == [
        ("model", "openai/clip-vit-base-patch32"),
        ("processor", "openai/clip-vit-base-patch32"),
    ]
    assert calls == [["blue shirt", "black shoes"], ["white skirt"]]
