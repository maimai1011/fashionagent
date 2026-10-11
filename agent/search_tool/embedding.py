
import asyncio
import logging
from threading import Lock

from sentence_transformers import SentenceTransformer

from search.schema import FashionItem


logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, model_name: str):
        logger.info("function_start function=EmbeddingService.__init__")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._lock = Lock()

        self.dimension = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        logger.info("function_start function=EmbeddingService._encode count=%d", len(texts))
        # Serialize access to the shared model for predictable
        # memory usage. Scale with separate workers if needed.
        with self._lock:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

    async def embed_items(
        self,
        items: list[FashionItem],
    ) -> list[list[float]]:
        logger.info("function_start function=EmbeddingService.embed_items count=%d", len(items))

        queries = []

        for item in items:
            attributes = [
                item.item_name,
                item.category,
                item.color,
                item.material,
                item.fit,
                item.styling_instruction,
            ]

            description = ", ".join(
                value for value in attributes if value
            )

            queries.append(description)

        # For models such as multilingual-e5,
        # the query prefix must match the indexing setup.
        return await asyncio.to_thread(
            self._encode,
            queries,
        )
