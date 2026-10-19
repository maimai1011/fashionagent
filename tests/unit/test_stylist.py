import logging

import pytest
from pydantic import ValidationError

from agent.agents.stylist.agent import PreservationError, StylistAgent
from agent.agents.stylist.schema import OutfitPlan, StylistRequest
from agent.agents.stylist.skill_loader import load_fashion_skill


def test_structured_output_requires_unique_ids(outfit):
    data = outfit.model_dump()
    data["items"][1]["outfit_item_id"] = "top-1"
    with pytest.raises(ValidationError):
        OutfitPlan.model_validate(data)


def test_skill_loading_is_complete_and_useful():
    skill = load_fashion_skill()
    assert "Color matching" in skill
    assert "Outfit composition" in skill
    assert "Occasion guidelines" in skill


@pytest.mark.asyncio
async def test_outfit_creation_with_mock_llm(outfit):
    captured = []

    def runner(messages):
        captured.extend(messages)
        return outfit.model_dump()

    agent = StylistAgent(runner=runner)
    result = await agent.run(StylistRequest(
        preferences={"occasion": "work"}, user_message="Suggest a work outfit",
    ))
    assert result == outfit
    assert '"user_message": "Suggest a work outfit"' in captured[0]["content"]


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
async def test_revision_rejects_changed_locked_item(outfit, caplog):
    invalid = outfit.model_copy(deep=True)
    invalid.items[1].color = "navy"
    agent = StylistAgent(runner=lambda _messages: invalid)
    with pytest.raises(PreservationError):
        await agent.run(StylistRequest(
            preferences={}, current_outfit=outfit,
            feedback="Change shirt", keep_categories=["bottom"],
        ))
    assert "stylist_inference_failed error_type=PreservationError" in caplog.text


@pytest.mark.asyncio
async def test_generic_locked_category_protects_specific_garment(outfit):
    invalid = outfit.model_copy(deep=True)
    invalid.items[1].color = "navy"
    agent = StylistAgent(runner=lambda _messages: invalid)

    with pytest.raises(PreservationError):
        await agent.run(StylistRequest(
            current_outfit=outfit, feedback="change the top",
            keep_categories=["bottom"],
        ))


@pytest.mark.asyncio
async def test_stylist_logs_summary_without_content_by_default(outfit, monkeypatch, caplog):
    monkeypatch.delenv("AGENT_LOG_CONTENT", raising=False)
    agent = StylistAgent(runner=lambda _messages: outfit)

    with caplog.at_level(logging.INFO):
        await agent.run(StylistRequest(user_message="private styling request"))

    assert "stylist_inference_start" in caplog.text
    assert "stylist_inference_complete" in caplog.text
    assert "private styling request" not in caplog.text
    assert "stylist_result" not in caplog.text


@pytest.mark.asyncio
async def test_stylist_can_log_request_and_result(outfit, monkeypatch, caplog):
    monkeypatch.setenv("AGENT_LOG_CONTENT", "true")
    agent = StylistAgent(runner=lambda _messages: outfit)

    with caplog.at_level(logging.INFO):
        await agent.run(StylistRequest(user_message="Suggest an outfit for work"))

    assert "stylist_request payload=" in caplog.text
    assert "Suggest an outfit for work" in caplog.text
    assert "stylist_result payload=" in caplog.text
    assert outfit.outfit_name in caplog.text
