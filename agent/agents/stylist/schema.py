"""Structured contracts for the independent stylist agent."""

import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

class OutfitItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outfit_item_id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=80)
    item_name: str = Field(min_length=1, max_length=200)
    color: str = Field(min_length=1, max_length=100)
    material: str = Field(min_length=1, max_length=100)
    fit: str = Field(min_length=1, max_length=100)
    styling_instruction: str = Field(min_length=1, max_length=500)


class OutfitPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outfit_name: str = Field(min_length=1, max_length=200)
    occasion: str = Field(min_length=1, max_length=100)
    style: str = Field(min_length=1, max_length=100)
    color_palette: list[str] = Field(min_length=1, max_length=8)
    items: list[OutfitItem] = Field(min_length=1, max_length=16)
    styling_tips: list[str] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def unique_item_ids(self) -> "OutfitPlan":
        logger.info("function_start function=OutfitPlan.unique_item_ids")
        ids = [item.outfit_item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("outfit_item_id values must be unique")
        return self


class WardrobeItemSummary(BaseModel):
    """One verified, completed upload available for styling."""

    item_id: str
    name: str
    category: str
    color: str


class StylistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferences: dict[str, object] = Field(default_factory=dict)
    user_message: str = ""
    current_outfit: OutfitPlan | None = None
    feedback: str = ""
    keep_categories: list[str] = Field(default_factory=list)
    keep_outfit_item_ids: list[str] = Field(default_factory=list)
    wardrobe_items: list[WardrobeItemSummary] = Field(default_factory=list)
    selected_wardrobe_items: dict[str, str] = Field(default_factory=dict)
