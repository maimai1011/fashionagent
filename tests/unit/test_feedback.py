from types import SimpleNamespace

import pytest

from agent.agents.intent.tools import build_tools
from agent.agents.stylist.schema import StylistRequest, WardrobeItemSummary
from agent.search.schema import OutfitItemMatch, ProductSearchResult
from agent.search.wardrobe import WardrobeInventory


class FakeStylist:
    def __init__(self, result):
        self.result = result
        self.request = None

    async def run(self, request: StylistRequest):
        self.request = request
        return self.result


class FakeSearch:
    def __init__(self, result):
        self.result = result
        self.request = None
        self.items = (
            WardrobeItemSummary(
                item_id="101", name="Oxford Shirt", category="shirt", color="white"
            ),
            WardrobeItemSummary(
                item_id="102", name="black trousers", category="pants", color="black"
            ),
        )

    async def inventory(self, user_id):
        return WardrobeInventory(user_id, self.items)

    async def search(self, request, inventory=None):
        self.request = request
        return self.result


def runtime(state):
    return SimpleNamespace(state=state, tool_call_id="call-1")


@pytest.mark.asyncio
async def test_partial_revision_keeps_unrelated_matches(outfit):
    revised = outfit.model_copy(deep=True)
    revised.items[0].color = "navy"
    state = {
        "current_outfit": outfit.model_dump(mode="json"), "outfit_version": 1,
        "outfit_history": [], "preferences": {}, "feedback_history": [],
        "product_matches": {"bottom-1": {"products": [{
            "product_id": "102", "name": "black trousers", "category": "pants",
            "color": "black", "similarity": 0.8,
        }]}},
        "selected_products": {},
        "user_id": "alice",
    }
    search = FakeSearch(ProductSearchResult(
        outfit_name=revised.outfit_name,
        matches=[OutfitItemMatch(
            outfit_item_id="top-1", requested_item=revised.items[0], products=[]
        )],
    ))
    tools = build_tools(FakeStylist(revised), search)
    command = await tools[1].coroutine(
        preferences={}, feedback="navy shirt", keep_categories=["bottom"],
        keep_outfit_item_ids=[], runtime=runtime(state),
    )
    assert search.request.outfit_item_ids == ["top-1"]
    assert command.update["product_matches"]["top-1"]["products"] == []
    assert command.update["product_matches"]["bottom-1"]["products"][0]["product_id"] == "102"
    assert command.update["outfit_version"] == 2


@pytest.mark.asyncio
async def test_product_alternative_search_preserves_outfit(outfit):
    search = FakeSearch(ProductSearchResult(outfit_name=outfit.outfit_name, matches=[]))
    state = {
        "user_id": "alice", "current_outfit": outfit.model_dump(mode="json"),
        "product_matches": {},
    }
    tools = build_tools(FakeStylist(outfit), search)
    command = await tools[2].coroutine(
        outfit_item_ids=["top-1"], color=None, excluded_item_ids=["101"],
        alternative_to_item_id="101", feedback="show another shirt", top_k=5,
        runtime=runtime(state),
    )
    assert search.request.constraints.excluded_product_ids == ["101"]
    assert search.request.alternative_to_product_id == "101"
    assert "current_outfit" not in command.update
    assert command.update["feedback_history"][0]["type"] == "wardrobe_search"


@pytest.mark.asyncio
async def test_select_product_rejects_unverified_id(outfit):
    state = {
        "product_matches": {
            "top-1": {"products": [{"product_id": "101", "name": "Verified"}]}
        },
        "selected_products": {},
    }
    tools = build_tools(FakeStylist(outfit), FakeSearch(None))
    with pytest.raises(ValueError, match="not present"):
        await tools[4].coroutine(
            outfit_item_id="top-1", item_id="invented", runtime=runtime(state)
        )


@pytest.mark.asyncio
async def test_restore_previous_version(outfit):
    previous = outfit.model_copy(deep=True)
    current = outfit.model_copy(deep=True)
    current.outfit_name = "Current"
    state = {
        "current_outfit": current.model_dump(mode="json"), "outfit_version": 2,
        "product_matches": {}, "selected_products": {},
        "outfit_history": [{
            "version": 1, "outfit": previous.model_dump(mode="json"),
            "product_matches": {"top-1": {"saved": True}}, "selected_products": {},
        }],
    }
    tools = build_tools(FakeStylist(current), FakeSearch(None))
    command = await tools[3].coroutine(runtime=runtime(state))
    assert command.update["outfit_version"] == 1
    assert command.update["product_matches"]["top-1"] == {"saved": True}
    assert command.update["outfit_history"] == []
