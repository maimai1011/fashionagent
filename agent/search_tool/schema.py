
from pydantic import BaseModel, Field, ConfigDict


class FashionItem(BaseModel):
    category: str
    item_name: str
    color: str | None = None
    material: str | None = None
    fit: str | None = None
    styling_instruction: str | None = None


class OutfitPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    outfit_name: str
    items: list[FashionItem] = Field(
        min_length=1,
        max_length=15,
    )


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    color: str | None = None
    material: str | None = None
    fit: str | None = None
    price: float | None = None
    image_url: str | None = None
    similarity: float
    ranking_score: float


class ItemMatches(BaseModel):
    requested_item: FashionItem
    products: list[Product]
    status: str


class SearchResponse(BaseModel):
    outfit_name: str
    matches: list[ItemMatches]
