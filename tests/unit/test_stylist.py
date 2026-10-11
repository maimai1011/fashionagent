import logging

import pytest
from pydantic import ValidationError

from agents.stylist.agent import PreservationError, StylistAgent
from agents.stylist.schema import OutfitPlan, StylistRequest
from agents.stylist.skill_loader import load_fashion_skill


def test_structured_output_requires_unique_ids(outfit):
    data = outfit.model_dump()
    data["items"][1]["outfit_item_id"] = "top-1"
    with pytest.raises(ValidationError):
        OutfitPlan.model_validate(data)


def test_skill_loading_includes_all_guidance(caplog):
    with caplog.at_level(logging.INFO):
        skill = load_fashion_skill()
    assert "Color matching" in skill
    assert "Outfit composition" in skill
    assert "Occasion guidelines" in skill
    assert "Constraints and revisions" in skill
    assert "function_start function=load_fashion_skill" in caplog.text


@pytest.mark.asyncio
async def test_outfit_creation_with_mock_llm(outfit):
    agent = StylistAgent(runner=lambda _messages: outfit.model_dump())
    result = await agent.run(StylistRequest(preferences={"occasion": "work"}))
    assert result == outfit


@pytest.mark.asyncio
async def test_revision_preserves_locked_item(outfit):
    revised = outfit.model_copy(deep=True)
    revised.items[0].color = "navy"
    agent = StylistAgent(runner=lambda _messages: revised)
    result = await agent.run(StylistRequest(
        preferences={}, current_outfit=outfit,
        feedback="Change shirt to navy", keep_categories=["bottom"],
    ))
    assert result.items[1] == outfit.items[1]


@pytest.mark.asyncio
async def test_revision_rejects_changed_locked_item(outfit):
    invalid = outfit.model_copy(deep=True)
    invalid.items[1].color = "navy"
    agent = StylistAgent(runner=lambda _messages: invalid)
    with pytest.raises(PreservationError):
        await agent.run(StylistRequest(
            preferences={}, current_outfit=outfit,
            feedback="Change shirt", keep_categories=["bottom"],
        ))
