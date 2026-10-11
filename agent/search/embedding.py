"""Asynchronous configurable query embedding provider."""

import asyncio
import logging
import os
import time
from typing import Protocol


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dimension: int
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedding:
    def __init__(self) -> None:
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
        self.normalize = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self._model = None
        self._lock = asyncio.Lock()
        self._encode_limit = asyncio.Semaphore(
            int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "2"))
        )

    async def _get_model(self):
        async with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
                actual = self._model.get_sentence_embedding_dimension()
                if actual != self.dimension:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: configured {self.dimension}, "
                        f"model {actual}. "
                        "Use the same model and dimension used to index Milvus."
                    )
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        model = await self._get_model()
        async with self._encode_limit:
            values = await asyncio.to_thread(
                model.encode,
                [self.query_prefix + text for text in texts],
                normalize_embeddings=self.normalize,
            )
        logger.info(
            "embedding_complete latency_ms=%.2f count=%d",
            (time.perf_counter() - started) * 1000,
            len(texts),
        )
        return [row.tolist() for row in values]
