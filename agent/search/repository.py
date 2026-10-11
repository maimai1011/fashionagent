"""Validated access to an existing Milvus product collection."""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.stylist.schema import OutfitItem
from search.ranking import normalized_similarity
from search.schema import MetricType, MilvusSchemaMapping, Product, ProductConstraints


logger = logging.getLogger(__name__)


class MilvusSchemaError(RuntimeError):
    pass


class MilvusProductRepository:
    """Search an existing collection without creating or modifying it."""

    def __init__(
        self,
        client: Any | None = None,
        mapping: MilvusSchemaMapping | None = None,
    ) -> None:
        logger.info("function_start function=MilvusProductRepository.__init__")
        self.collection = os.getenv("MILVUS_PRODUCT_COLLECTION", "products")
        self.mapping = mapping or MilvusSchemaMapping()
        self.timeout = float(os.getenv("MILVUS_SEARCH_TIMEOUT_SECONDS", "8"))
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self.configured_metric = os.getenv("MILVUS_METRIC_TYPE", "COSINE").upper()
        if self.configured_metric not in {"COSINE", "IP", "L2"}:
            raise ValueError("MILVUS_METRIC_TYPE must be COSINE, IP, or L2")
        self.metric: MetricType = self.configured_metric  # type: ignore[assignment]
        self._client = client
        self._validated = False
        self._validation_lock = asyncio.Lock()
        self._stock_is_bool = False
        self._product_id_is_string = True
        self._executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("MILVUS_MAX_CONCURRENCY", "4")),
            thread_name_prefix="milvus-search",
        )

    def _get_client(self) -> Any:
        logger.info("function_start function=MilvusProductRepository._get_client")
        if self._client is None:
            from pymilvus import MilvusClient
            uri = os.getenv("MILVUS_URI")
            if not uri:
                host = os.getenv("MILVUS_HOST", "localhost")
                port = os.getenv("MILVUS_PORT", "19530")
                uri = f"http://{host}:{port}"
            kwargs: dict[str, Any] = {"uri": uri}
            if token := os.getenv("MILVUS_TOKEN"):
                kwargs["token"] = token
            self._client = MilvusClient(**kwargs)
        return self._client

    async def validate(self) -> None:
        logger.info("function_start function=MilvusProductRepository.validate")
        async with self._validation_lock:
            if self._validated:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._validate_sync)

    def _validate_sync(self) -> None:
        logger.info("function_start function=MilvusProductRepository._validate_sync")
        client = self._get_client()
        if not client.has_collection(collection_name=self.collection):
            raise MilvusSchemaError(
                f"Milvus collection '{self.collection}' does not exist. Set "
                "MILVUS_PRODUCT_COLLECTION to the existing catalog collection."
            )
        description = client.describe_collection(collection_name=self.collection)
        fields = {field["name"]: field for field in description.get("fields", [])}
        missing = self.mapping.required_fields() - fields.keys()
        if missing:
            raise MilvusSchemaError(
                f"Collection '{self.collection}' is incompatible; missing fields: "
                f"{sorted(missing)}. Configure MILVUS_FIELD_* mappings."
            )
        vector = fields[self.mapping.vector]
        from pymilvus import DataType
        vector_type = DataType(vector["type"])
        if vector_type != DataType.FLOAT_VECTOR:
            raise MilvusSchemaError(
                f"Configured vector field '{self.mapping.vector}' is "
                f"{vector_type.name}, not FLOAT_VECTOR"
            )
        product_id_type = DataType(fields[self.mapping.product_id]["type"])
        if product_id_type not in {DataType.VARCHAR, DataType.INT64}:
            raise MilvusSchemaError("Product ID field must be VARCHAR or INT64")
        self._product_id_is_string = product_id_type == DataType.VARCHAR
        text_fields = {
            self.mapping.name,
            self.mapping.description,
            self.mapping.category,
            self.mapping.color,
            self.mapping.material,
            self.mapping.fit,
            self.mapping.image_url,
        }
        invalid_text = [
            name for name in text_fields
            if DataType(fields[name]["type"]) != DataType.VARCHAR
        ]
        if invalid_text:
            raise MilvusSchemaError(
                f"Configured text fields are not VARCHAR: {sorted(invalid_text)}"
            )
        price_type = DataType(fields[self.mapping.price]["type"])
        price_types = {
            DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
            DataType.FLOAT, DataType.DOUBLE,
        }
        if price_type not in price_types:
            raise MilvusSchemaError("Price field must be an integer, FLOAT, or DOUBLE")
        stock_type = DataType(fields[self.mapping.stock]["type"])
        numeric_types = {DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64}
        if stock_type not in numeric_types | {DataType.BOOL}:
            raise MilvusSchemaError("Stock field must be BOOL or an integer type")
        self._stock_is_bool = stock_type == DataType.BOOL
        params = vector.get("params", {})
        actual_dim = int(params.get("dim", vector.get("dim", 0)))
        if actual_dim != self.dimension:
            raise MilvusSchemaError(
                f"Vector dimension mismatch: collection={actual_dim}, embedding={self.dimension}"
            )
        index_names = client.list_indexes(
            collection_name=self.collection,
            field_name=self.mapping.vector,
        )
        if not index_names:
            raise MilvusSchemaError(f"Vector field '{self.mapping.vector}' has no index")
        index = client.describe_index(
            collection_name=self.collection,
            index_name=index_names[0],
        )
        actual_metric = str(index.get("metric_type", "")).upper()
        if actual_metric and actual_metric != self.metric:
            raise MilvusSchemaError(
                f"Metric mismatch: collection={actual_metric}, configured={self.metric}"
            )
        client.load_collection(collection_name=self.collection)
        self._validated = True

    @staticmethod
    def _literal(value: str) -> str:
        logger.info("function_start function=MilvusProductRepository._literal")
        return json.dumps(value, ensure_ascii=False)

    def _filter(self, item: OutfitItem, constraints: ProductConstraints) -> str:
        logger.info(
            "function_start function=MilvusProductRepository._filter item_id=%s",
            item.outfit_item_id,
        )
        m = self.mapping
        clauses = [f"{m.category} == {self._literal(item.category)}"]
        if constraints.color:
            clauses.append(f"{m.color} == {self._literal(constraints.color)}")
        for material in constraints.excluded_materials:
            clauses.append(f"{m.material} != {self._literal(material)}")
        if constraints.max_price is not None:
            clauses.append(f"{m.price} <= {float(constraints.max_price)}")
        if constraints.in_stock_only:
            clauses.append(f"{m.stock} == true" if self._stock_is_bool else f"{m.stock} > 0")
        if constraints.excluded_product_ids:
            if self._product_id_is_string:
                values = ", ".join(self._literal(x) for x in constraints.excluded_product_ids)
            else:
                try:
                    values = ", ".join(str(int(x)) for x in constraints.excluded_product_ids)
                except ValueError as exc:
                    raise ValueError(
                        "Excluded product IDs must be integers for this collection"
                    ) from exc
            clauses.append(f"{m.product_id} not in [{values}]")
        return " and ".join(clauses)

    async def search(
        self,
        item: OutfitItem,
        vector: list[float],
        constraints: ProductConstraints,
        top_k: int,
    ) -> list[Product]:
        logger.info(
            "function_start function=MilvusProductRepository.search item_id=%s top_k=%d",
            item.outfit_item_id,
            top_k,
        )
        if not self._validated:
            await self.validate()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            self._search_sync,
            item,
            vector,
            constraints,
            top_k,
        )
        return await asyncio.wait_for(future, timeout=self.timeout + 1)

    def _search_sync(
        self,
        item: OutfitItem,
        vector: list[float],
        constraints: ProductConstraints,
        top_k: int,
    ) -> list[Product]:
        logger.info(
            "function_start function=MilvusProductRepository._search_sync item_id=%s",
            item.outfit_item_id,
        )
        started = time.perf_counter()
        m = self.mapping
        output_fields = [
            m.product_id, m.name, m.description, m.category, m.color,
            m.material, m.fit, m.price, m.stock, m.image_url,
        ]
        raw = self._get_client().search(
            collection_name=self.collection,
            data=[vector],
            anns_field=m.vector,
            filter=self._filter(item, constraints),
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": self.metric, "params": {}},
            timeout=self.timeout,
        )
        hits = raw[0] if raw else []
        products: list[Product] = []
        for hit in hits:
            entity = hit.get("entity", hit)
            products.append(Product(
                product_id=str(entity[m.product_id]),
                name=str(entity[m.name]),
                description=str(entity.get(m.description, "")),
                category=str(entity[m.category]),
                color=str(entity.get(m.color, "")),
                material=str(entity.get(m.material, "")),
                fit=str(entity.get(m.fit, "")),
                price=float(entity[m.price]),
                stock=entity[m.stock],
                image_url=entity.get(m.image_url) or None,
                similarity=normalized_similarity(float(hit["distance"]), self.metric),
            ))
        logger.info(
            "milvus_search_complete latency_ms=%.2f results=%d item_id=%s",
            (time.perf_counter() - started) * 1000,
            len(products), item.outfit_item_id,
        )
        return products

    def close(self) -> None:
        logger.info("function_start function=MilvusProductRepository.close")
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
