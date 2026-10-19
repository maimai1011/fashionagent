"""Search uploaded wardrobe vectors and resolve their PostgreSQL metadata."""

import asyncio
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agent.agents.stylist.schema import OutfitItem, WardrobeItemSummary
from agent.observability.context import celery_headers
from agent.search.categories import (
    category_compatible,
    normalize_category,
    normalize_color,
    wardrobe_display_name,
)
from agent.search.ranking import normalized_similarity
from agent.search.repository import MilvusSchemaError
from agent.search.schema import (
    OutfitItemMatch,
    Product,
    ProductConstraints,
    ProductSearchResult,
    SearchRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WardrobeInventory:
    """Completed uploads fetched for one user during a styling turn."""

    user_id: str
    items: tuple[WardrobeItemSummary, ...]


class ClipTextEmbedding:
    """Ask the image worker to encode text with its existing CLIP model."""

    dimension = 512

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(self._embed_sync, texts)
        if len(vectors) != len(texts) or any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("CLIP worker returned an unexpected embedding shape")
        return vectors

    @staticmethod
    def _embed_sync(texts: list[str]) -> list[list[float]]:
        from inference.image_inference.tasks import embed_wardrobe_queries

        task = embed_wardrobe_queries.apply_async(args=[texts], headers=celery_headers())
        return task.get(timeout=float(os.getenv("WARDROBE_EMBEDDING_TIMEOUT_SECONDS", "120")))


class MilvusWardrobeRepository:
    """Search only completed uploads belonging to the current user."""

    metric = "COSINE"

    def __init__(self, client=None) -> None:
        self.collection = os.getenv("MILVUS_COLLECTION", "image_features")
        self.dimension = int(os.getenv("MILVUS_EMBEDDING_DIM", "512"))
        self.timeout = float(os.getenv("MILVUS_SEARCH_TIMEOUT_SECONDS", "8"))
        self.min_similarity = float(os.getenv("WARDROBE_MIN_SIMILARITY", "0.20"))
        if not math.isfinite(self.min_similarity) or not 0 <= self.min_similarity <= 1:
            raise ValueError("WARDROBE_MIN_SIMILARITY must be between 0 and 1")
        self._client = client
        self._loaded = False
        self._load_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("MILVUS_MAX_CONCURRENCY", "4")),
            thread_name_prefix="milvus-wardrobe",
        )

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            uri = os.getenv("MILVUS_URI") or (
                f"http://{os.getenv('MILVUS_HOST', 'localhost')}:"
                f"{os.getenv('MILVUS_PORT', '19530')}"
            )
            token = os.getenv("MILVUS_TOKEN")
            self._client = (
                MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
            )
        return self._client

    async def validate(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._validate_sync)

    def _validate_sync(self) -> None:
        from pymilvus import DataType

        client = self._get_client()
        if not client.has_collection(collection_name=self.collection):
            raise MilvusSchemaError(
                f"Milvus collection '{self.collection}' does not exist. "
                "Run the storage initializer before searching uploaded images."
            )
        fields = {
            field["name"]: field
            for field in client.describe_collection(collection_name=self.collection)["fields"]
        }
        if not {"image_name", "embedding"} <= fields.keys():
            raise MilvusSchemaError(
                f"Milvus collection '{self.collection}' needs image_name and embedding fields"
            )
        vector = fields["embedding"]
        if DataType(vector["type"]) != DataType.FLOAT_VECTOR:
            raise MilvusSchemaError("Wardrobe embedding must be FLOAT_VECTOR")
        actual_dim = int(vector["params"]["dim"])
        if actual_dim != self.dimension:
            raise MilvusSchemaError(
                f"Wardrobe embedding dimension is {actual_dim}, expected {self.dimension}"
            )
        self._ensure_loaded_sync()

    def _ensure_loaded_sync(self) -> None:
        with self._load_lock:
            if not self._loaded:
                self._get_client().load_collection(collection_name=self.collection)
                self._loaded = True

    @staticmethod
    def _get_connection():
        import psycopg2

        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.environ["POSTGRES_PORT"],
        )

    async def count_items(self, user_id: str) -> int:
        """Count this user's completed wardrobe items."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._count_items_sync, user_id)

    @classmethod
    def _count_items_sync(cls, user_id: str) -> int:
        connection = cls._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM wardrobe_items
                    WHERE user_id = %s AND status = 'completed'
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
        finally:
            connection.close()
        return int(row[0])

    async def inventory(self, user_id: str) -> WardrobeInventory:
        """Read completed uploads once for the current user."""
        if not user_id.strip():
            raise ValueError("user_id is required to load wardrobe items")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._inventory_sync, user_id)

    @classmethod
    def _inventory_sync(cls, user_id: str) -> WardrobeInventory:
        connection = cls._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT item_id, original_filename, clothing_type, dominant_color
                    FROM wardrobe_items
                    WHERE user_id = %s AND status = 'completed'
                    ORDER BY created_at DESC NULLS LAST, item_id DESC
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()
        finally:
            connection.close()

        return WardrobeInventory(
            user_id=user_id,
            items=tuple(
                WardrobeItemSummary(
                    item_id=str(item_id), name=filename,
                    category=clothing_type or "", color=dominant_color or "",
                )
                for item_id, filename, clothing_type, dominant_color in rows
            ),
        )

    @staticmethod
    def eligible_items(
        inventory: WardrobeInventory,
        item: OutfitItem,
        constraints: ProductConstraints,
    ) -> dict[str, WardrobeItemSummary]:
        """Apply ownership, explicit constraints, and garment compatibility before Milvus."""
        if constraints.excluded_materials:
            raise ValueError("Uploaded wardrobe items do not have material metadata")
        excluded = set(constraints.excluded_product_ids)
        color = normalize_color(constraints.color) if constraints.color else None
        return {
            candidate.item_id: candidate
            for candidate in inventory.items
            if candidate.item_id not in excluded
            and (color is None or normalize_color(candidate.color) == color)
            and category_compatible(
                item.category, item.item_name, candidate.category, candidate.name
            )
        }

    async def search(
        self,
        item: OutfitItem,
        vector: list[float],
        constraints: ProductConstraints,
        top_k: int,
        user_id: str,
        inventory: WardrobeInventory | None = None,
    ) -> list[Product]:
        inventory = inventory or await self.inventory(user_id)
        if inventory.user_id != user_id:
            raise ValueError("Wardrobe inventory belongs to a different user")
        eligible = self.eligible_items(inventory, item, constraints)
        return await self.search_prepared(item, vector, top_k, eligible)

    async def search_prepared(
        self,
        item: OutfitItem,
        vector: list[float],
        top_k: int,
        eligible: dict[str, WardrobeItemSummary],
    ) -> list[Product]:
        if len(vector) != self.dimension:
            raise ValueError(f"CLIP query vector must have {self.dimension} values")
        if not eligible:
            return []
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            self._search_prepared_sync,
            item,
            vector,
            top_k,
            eligible,
        )
        return await asyncio.wait_for(future, timeout=self.timeout + 10)

    def _search_sync(
        self,
        item: OutfitItem,
        vector: list[float],
        constraints: ProductConstraints,
        top_k: int,
        user_id: str,
    ) -> list[Product]:
        """Compatibility entry point for direct repository callers."""
        inventory = self._inventory_sync(user_id)
        eligible = self.eligible_items(inventory, item, constraints)
        return self._search_prepared_sync(item, vector, top_k, eligible)

    def _search_prepared_sync(
        self,
        item: OutfitItem,
        vector: list[float],
        top_k: int,
        eligible: dict[str, WardrobeItemSummary],
    ) -> list[Product]:
        started = time.perf_counter()
        if not eligible:
            logger.info(
                "wardrobe_search_skipped item_id=%s reason=no_eligible_category",
                item.outfit_item_id,
            )
            return []
        self._ensure_loaded_sync()
        hits = self._get_client().search(
            collection_name=self.collection,
            data=[vector],
            anns_field="embedding",
            filter=f"image_name in {json.dumps(list(eligible))}",
            limit=min(len(eligible), max(top_k * 5, 25), 150),
            output_fields=["image_name"],
            search_params={"metric_type": self.metric, "params": {}},
            timeout=self.timeout,
        )
        base_url = os.getenv("API_BASE_URL") or (
            f"http://{os.getenv('API_HOST', '127.0.0.1')}:"
            f"{os.getenv('API_PORT', '8000')}"
        )
        ranked: list[tuple[float, Product]] = []
        best_hits: dict[str, float] = {}
        rejected_similarity = 0
        rejected_ownership = 0
        duplicates = 0
        for hit in hits[0] if hits else []:
            entity = hit.get("entity", hit)
            item_id = str(entity["image_name"])
            if item_id not in eligible:
                rejected_ownership += 1
                continue
            similarity = normalized_similarity(float(hit["distance"]), self.metric)
            if not math.isfinite(similarity):
                rejected_similarity += 1
                continue
            if item_id in best_hits:
                duplicates += 1
                best_hits[item_id] = max(best_hits[item_id], similarity)
            else:
                best_hits[item_id] = similarity
        for item_id, similarity in best_hits.items():
            if similarity < self.min_similarity:
                rejected_similarity += 1
                continue
            candidate = eligible[item_id]
            category = normalize_category(candidate.category, candidate.name)
            product = Product(
                product_id=item_id,
                name=wardrobe_display_name(candidate.category, candidate.color, candidate.name),
                description="Uploaded wardrobe item",
                category=category or candidate.category,
                color=candidate.color,
                stock=None,
                image_url=f"{base_url.rstrip('/')}/v1/wardrobe/{item_id}/image",
                similarity=similarity,
            )
            exact_type = category == normalize_category(item.category, item.item_name)
            exact_color = normalize_color(candidate.color) == normalize_color(item.color)
            score = similarity + (0.04 if exact_type else 0.0) + (0.015 if exact_color else 0.0)
            ranked.append((score, product))
        ranked.sort(key=lambda entry: (-entry[0], -entry[1].similarity, entry[1].product_id))
        products = [product for _, product in ranked[:top_k]]
        logger.info(
            "wardrobe_search_complete item_id=%s eligible=%d hits=%d results=%d "
            "low_similarity=%d rejected_ownership=%d duplicates=%d threshold=%.3f latency_ms=%.2f",
            item.outfit_item_id,
            len(eligible),
            len(hits[0]) if hits else 0,
            len(products),
            rejected_similarity,
            rejected_ownership,
            duplicates,
            self.min_similarity,
            (time.perf_counter() - started) * 1000,
        )
        return products

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._executor.shutdown(wait=False, cancel_futures=True)


class WardrobeSearchService:
    """Match outfit descriptions to the current user's processed uploads."""

    def __init__(
        self,
        repository: MilvusWardrobeRepository | None = None,
        embedding: ClipTextEmbedding | None = None,
    ) -> None:
        self.repository = repository or MilvusWardrobeRepository()
        self.embedding = embedding or ClipTextEmbedding()

    @property
    def min_similarity(self) -> float:
        return self.repository.min_similarity

    async def count_items(self, user_id: str) -> int:
        """Return the number of completed items in the user's wardrobe."""
        if not user_id.strip():
            raise ValueError("user_id is required to count wardrobe items")
        return await self.repository.count_items(user_id)

    async def inventory(self, user_id: str) -> WardrobeInventory:
        """Expose a single completed-item snapshot to styling and search."""
        return await self.repository.inventory(user_id)

    @staticmethod
    def _clip_query(item: OutfitItem) -> str:
        kind = normalize_category(item.category, item.item_name) or item.category
        color = normalize_color(item.color)
        plural = {"shoes", "jeans", "pants", "shorts", "jewelry", "activewear"}
        article = "" if kind in plural else "a "
        return f"a photo of {article}{color} {kind}, {kind} clothing"

    async def search(
        self,
        request: SearchRequest,
        inventory: WardrobeInventory | None = None,
    ) -> ProductSearchResult:
        if not request.user_id.strip():
            raise ValueError("user_id is required to search uploaded wardrobe items")
        inventory = inventory or await self.inventory(request.user_id)
        if inventory.user_id != request.user_id:
            raise ValueError("Wardrobe inventory belongs to a different user")
        selected = [
            item for item in request.outfit.items
            if not request.outfit_item_ids or item.outfit_item_id in request.outfit_item_ids
        ]
        if not selected:
            raise ValueError("No outfit items match outfit_item_ids")
        constraints = request.constraints.model_copy(deep=True)
        if request.alternative_to_product_id:
            constraints.excluded_product_ids.append(request.alternative_to_product_id)
        prepared = [
            self.repository.eligible_items(inventory, item, constraints)
            for item in selected
        ]
        queries = list(dict.fromkeys(
            self._clip_query(item) for item, eligible in zip(selected, prepared, strict=True)
            if eligible
        ))
        vectors = await self.embedding.embed(queries) if queries else []
        query_vectors = dict(zip(queries, vectors, strict=True))
        matches = await asyncio.gather(*(
            self.repository.search_prepared(
                item, query_vectors[self._clip_query(item)], request.top_k, eligible
            ) if eligible else asyncio.sleep(0, result=[])
            for item, eligible in zip(selected, prepared, strict=True)
        ))
        logger.info(
            "wardrobe_retrieval_summary outfit_items=%d inventory=%d searchable=%d "
            "embedding_queries=%d results=%d",
            len(selected), len(inventory.items), sum(bool(items) for items in prepared),
            len(queries), sum(len(items) for items in matches),
        )
        return ProductSearchResult(
            outfit_name=request.outfit.outfit_name,
            matches=[
                OutfitItemMatch(
                    outfit_item_id=item.outfit_item_id,
                    requested_item=item,
                    products=products,
                )
                for item, products in zip(selected, matches, strict=True)
            ],
        )

    async def validate(self) -> None:
        await self.repository.validate()

    def close(self) -> None:
        self.repository.close()
