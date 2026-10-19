
from __future__ import annotations

import asyncio
import logging
import math

from pymilvus import MilvusClient

from search.embedding import EmbeddingService
from search.schema import (
    FashionItem,
    ItemMatches,
    OutfitPlan,
    Product,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class FashionSearchService:

    OUTPUT_FIELDS = [
        "product_id",
        "name",
        "category",
        "color",
        "material",
        "fit",
        "image_url",
    ]

    def __init__(
        self,
        milvus_uri: str,
        collection_name: str,
        embedding_model: str,
        milvus_token: str | None = None,
        vector_field: str = "embedding",
        metric_type: str = "COSINE",
        top_k: int = 5,
        candidate_limit: int = 30,
        max_concurrency: int = 4,
    ):
        if metric_type != "COSINE":
            raise ValueError(
                "This ranking implementation requires COSINE"
            )

        if not (1 <= top_k <= candidate_limit):
            raise ValueError("Invalid top_k or candidate_limit")

        self.collection_name = collection_name
        self.vector_field = vector_field
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.semaphore = asyncio.Semaphore(max_concurrency)

        connection = {"uri": milvus_uri}

        if milvus_token:
            connection["token"] = milvus_token

        self.client = MilvusClient(**connection)

        self.embedder = EmbeddingService(
            model_name=embedding_model
        )

        self.search_params = {
            "metric_type": metric_type,
            "params": {},
        }

    def initialize(self):
        """Validate the collection and load it for search."""

        schema = self.client.describe_collection(
            collection_name=self.collection_name
        )

        fields = {
            field["name"]: field
            for field in schema["fields"]
        }

        required = set(self.OUTPUT_FIELDS)
        required.add(self.vector_field)
        required.add("in_stock")

        missing = required - fields.keys()

        if missing:
            raise RuntimeError(
                f"Missing Milvus fields: {sorted(missing)}"
            )

        vector_info = fields[self.vector_field]

        actual_dim = int(
            vector_info.get("params", {}).get("dim", 0)
        )

        if (
            actual_dim
            and self.embedder.dimension
            and actual_dim != self.embedder.dimension
        ):
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"Milvus={actual_dim}, "
                f"Model={self.embedder.dimension}"
            )

        self.client.load_collection(
            collection_name=self.collection_name
        )

        logger.info(
            "Milvus collection ready: %s",
            self.collection_name,
        )

    @staticmethod
    def _matches(expected: str | None, actual: str | None):
        if not expected or not actual:
            return False

        return expected.strip().casefold() == actual.strip().casefold()

    def _rank(
        self,
        item: FashionItem,
        hits: list[dict],
    ) -> list[Product]:

        ranked = []

        for hit in hits:
            entity = hit.get("entity", {})

            # For COSINE, higher scores are more similar.
            similarity = float(hit["distance"])

            if not math.isfinite(similarity):
                continue

            # Metadata bonuses are tuning parameters,
            # not calibrated relevance probabilities.
            bonus = 0.0

            if self._matches(
                item.color,
                entity.get("color"),
            ):
                bonus += 0.12

            if self._matches(
                item.material,
                entity.get("material"),
            ):
                bonus += 0.06

            if self._matches(
                item.fit,
                entity.get("fit"),
            ):
                bonus += 0.06

            ranked.append(
                Product(
                    product_id=str(entity["product_id"]),
                    name=entity["name"],
                    category=entity["category"],
                    color=entity.get("color"),
                    material=entity.get("material"),
                    fit=entity.get("fit"),
                    image_url=entity.get("image_url"),
                    similarity=similarity,
                    ranking_score=similarity + bonus,
                )
            )

        ranked.sort(
            key=lambda p: p.ranking_score,
            reverse=True,
        )

        return ranked[:self.top_k]

    def _search_sync(
        self,
        item: FashionItem,
        vector: list[float],
    ) -> list[Product]:

        # Parameterized filter: user text is not
        # interpolated into the Milvus expression.
        expression = (
            "category == {category} "
            "and in_stock == true"
        )

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=self.vector_field,
            filter=expression,
            filter_params={
                "category": item.category,
            },
            limit=self.candidate_limit,
            output_fields=self.OUTPUT_FIELDS,
            search_params=self.search_params,
            timeout=10,
        )

        hits = results[0] if results else []

        return self._rank(item, hits)

    async def search_item(
        self,
        item: FashionItem,
        vector: list[float],
    ) -> ItemMatches:

        async with self.semaphore:
            products = await asyncio.to_thread(
                self._search_sync,
                item,
                vector,
            )

        return ItemMatches(
            requested_item=item,
            products=products,
            status="matched" if products else "no_match",
        )

    async def search_outfit(
        self,
        outfit: OutfitPlan,
    ) -> SearchResponse:

        vectors = await self.embedder.embed_items(
            outfit.items
        )

        tasks = [
            self.search_item(item, vector)
            for item, vector in zip(
                outfit.items,
                vectors,
                strict=True,
            )
        ]

        # Database failures propagate rather than
        # being disguised as "no products found".
        matches = await asyncio.gather(*tasks)

        return SearchResponse(
            outfit_name=outfit.outfit_name,
            matches=matches,
        )

    def close(self):
        self.client.close()
