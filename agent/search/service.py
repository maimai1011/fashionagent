"""Concurrent semantic product search with stable outfit-item association."""

import asyncio
import logging
import os

from agent.agents.stylist.schema import OutfitItem
from agent.search.embedding import EmbeddingProvider, SentenceTransformerEmbedding
from agent.search.ranking import rerank
from agent.search.repository import MilvusProductRepository
from agent.search.schema import OutfitItemMatch, ProductSearchResult, SearchRequest

logger = logging.getLogger(__name__)


class ProductSearchService:
    def __init__(
        self,
        repository: MilvusProductRepository | None = None,
        embedding: EmbeddingProvider | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        logger.info("function_start function=ProductSearchService.__init__")
        self.repository = repository or MilvusProductRepository()
        self.embedding = embedding or SentenceTransformerEmbedding()
        concurrency = max_concurrency or int(os.getenv("MILVUS_MAX_CONCURRENCY", "4"))
        self._semaphore = asyncio.Semaphore(concurrency)

    @staticmethod
    def _query(item: OutfitItem) -> str:
        logger.info("function_start function=ProductSearchService._query")
        return (
            f"{item.category}: {item.item_name}; color {item.color}; "
            f"material {item.material}; fit {item.fit}"
        )

    async def search(self, request: SearchRequest) -> ProductSearchResult:
        logger.info(
            "function_start function=ProductSearchService.search item_count=%d",
            len(request.outfit.items),
        )
        selected = [
            item for item in request.outfit.items
            if not request.outfit_item_ids or item.outfit_item_id in request.outfit_item_ids
        ]
        if not selected:
            raise ValueError("No outfit items match outfit_item_ids")
        constraints = request.constraints.model_copy(deep=True)
        if request.alternative_to_product_id:
            constraints.excluded_product_ids.append(request.alternative_to_product_id)
        vectors = await self.embedding.embed([self._query(item) for item in selected])

        async def one(item: OutfitItem, vector: list[float]) -> OutfitItemMatch:
            logger.info(
                "function_start function=ProductSearchService.search.one item_id=%s",
                item.outfit_item_id,
            )
            async with self._semaphore:
                products = await self.repository.search(item, vector, constraints, request.top_k)
                return OutfitItemMatch(
                    outfit_item_id=item.outfit_item_id,
                    requested_item=item,
                    products=rerank(products, self.repository.metric),
                )

        tasks = (one(item, vector) for item, vector in zip(selected, vectors, strict=True))
        matches = await asyncio.gather(*tasks)
        return ProductSearchResult(outfit_name=request.outfit.outfit_name, matches=matches)

    async def validate(self) -> None:
        logger.info("function_start function=ProductSearchService.validate")
        await self.repository.validate()

    def close(self) -> None:
        logger.info("function_start function=ProductSearchService.close")
        self.repository.close()
