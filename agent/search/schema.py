"""Product search contracts and configurable Milvus field mapping."""

import os
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.agents.stylist.schema import OutfitItem, OutfitPlan

DEFAULT_SEARCH_TOP_K = 5
MAX_SEARCH_TOP_K = 5


class MilvusSchemaMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = os.getenv("MILVUS_FIELD_PRODUCT_ID", "product_id")
    vector: str = os.getenv("MILVUS_FIELD_VECTOR", "embedding")
    name: str = os.getenv("MILVUS_FIELD_NAME", "name")
    description: str = os.getenv("MILVUS_FIELD_DESCRIPTION", "description")
    category: str = os.getenv("MILVUS_FIELD_CATEGORY", "category")
    color: str = os.getenv("MILVUS_FIELD_COLOR", "color")
    material: str = os.getenv("MILVUS_FIELD_MATERIAL", "material")
    fit: str = os.getenv("MILVUS_FIELD_FIT", "fit")
    price: str = os.getenv("MILVUS_FIELD_PRICE", "price")
    stock: str = os.getenv("MILVUS_FIELD_STOCK", "stock")
    image_url: str = os.getenv("MILVUS_FIELD_IMAGE_URL", "image_url")

    @field_validator("*")
    @classmethod
    def valid_identifier(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError(f"invalid Milvus field identifier: {value}")
        return value

    def required_fields(self) -> set[str]:
        return set(self.model_dump().values())


class ProductConstraints(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    color: str | None = Field(default=None, max_length=100)
    excluded_materials: list[str] = Field(default_factory=list, max_length=20)
    max_price: float | None = Field(default=None, ge=0)
    excluded_product_ids: list[str] = Field(default_factory=list, max_length=100)
    in_stock_only: bool = True


class Product(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    product_id: str
    name: str
    description: str = ""
    category: str
    color: str = ""
    material: str = ""
    fit: str = ""
    price: float | None = Field(default=None, ge=0)
    stock: int | bool | None = None
    image_url: str | None = None
    similarity: float

    @field_validator("image_url")
    @classmethod
    def valid_image_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image_url must be an absolute HTTP(S) URL")
        return value


class OutfitItemMatch(BaseModel):
    outfit_item_id: str
    requested_item: OutfitItem
    products: list[Product]


class ProductSearchResult(BaseModel):
    outfit_name: str
    matches: list[OutfitItemMatch]


class SearchRequest(BaseModel):
    outfit: OutfitPlan
    user_id: str = ""
    outfit_item_ids: list[str] = Field(default_factory=list)
    constraints: ProductConstraints = Field(default_factory=ProductConstraints)
    top_k: int = Field(
        default=DEFAULT_SEARCH_TOP_K,
        ge=DEFAULT_SEARCH_TOP_K,
        le=MAX_SEARCH_TOP_K,
    )
    alternative_to_product_id: str | None = None


MetricType = Literal["COSINE", "IP", "L2"]
