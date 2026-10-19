from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent.agents.intent.tools import build_tools
from agent.agents.stylist.schema import StylistRequest, WardrobeItemSummary
from agent.search.schema import OutfitItemMatch, Product, ProductSearchResult
from agent.search.wardrobe import WardrobeInventory


class FakeStylist:
    def __init__(self, result):
        self.result = result
        self.request = None

    async def run(self, request: StylistRequest):
        self.request = request
        return self.result


class FakeSearch:
    def __init__(self, result, items=None):
        self.result = result
        self.request = None
        self.items = items if items is not None else (
            WardrobeItemSummary(
                item_id="101", name="Oxford Shirt", category="shirt", color="white"
            ),
            WardrobeItemSummary(
                item_id="102", name="black trousers", category="pants", color="black"
            ),
        )

    async def inventory(self, user_id):
        self.inventory_user_id = user_id
        self.inventory_calls = getattr(self, "inventory_calls", 0) + 1
        return WardrobeInventory(user_id, tuple(self.items))

    async def search(self, request, inventory=None):
        self.request = request
        self.search_inventory = inventory
        return ProductSearchResult(
            outfit_name=self.result.outfit_name,
            matches=[
                match for match in self.result.matches
                if not request.outfit_item_ids or match.outfit_item_id in request.outfit_item_ids
            ],
        )

    async def count_items(self, user_id):
        self.user_id = user_id
        return 2


def runtime(state):
    return SimpleNamespace(state=state, tool_call_id="call-1")


@pytest.mark.asyncio
async def test_count_wardrobe_works_without_an_outfit(outfit):
    search = FakeSearch(None)
    tools = build_tools(FakeStylist(outfit), search)
    command = await tools[5].coroutine(runtime=runtime({"user_id": "alice"}))
    assert search.user_id == "alice"
    assert '"count": 2' in command.update["messages"][0].content
    assert '"status": "completed"' in command.update["messages"][0].content
    assert "current_outfit" not in command.update


@pytest.mark.asyncio
async def test_partial_revision_reuses_unchanged_item_and_searches_changed_item(outfit):
    revised = outfit.model_copy(deep=True)
    revised.items[0].color = "navy"
    state = {
        "current_outfit": outfit.model_dump(mode="json"), "outfit_version": 1,
        "outfit_history": [], "preferences": {}, "feedback_history": [],
        "product_matches": {
            "top-1": {"old": "top"},
            "bottom-1": {"products": [{
                "product_id": "102", "name": "black trousers", "category": "pants",
                "color": "black", "similarity": 0.8,
            }]},
        },
        "selected_products": {"bottom-1": {
            "product_id": "102", "name": "black trousers", "category": "pants",
            "color": "black", "similarity": 0.8,
        }},
        "user_id": "alice",
    }
    search = FakeSearch(ProductSearchResult(
        outfit_name=revised.outfit_name,
        matches=[
            OutfitItemMatch(outfit_item_id=item.outfit_item_id, requested_item=item, products=[])
            for item in revised.items
        ],
    ))
    tools = build_tools(FakeStylist(revised), search)
    command = await tools[1].coroutine(
        preferences={}, feedback="navy shirt", keep_categories=["bottom"],
        keep_outfit_item_ids=[], runtime=runtime(state),
    )
    assert search.request.outfit == revised
    assert search.request.outfit_item_ids == ["top-1"]
    assert search.search_inventory.user_id == "alice"
    assert set(command.update["product_matches"]) == {"top-1", "bottom-1"}
    assert command.update["product_matches"]["top-1"]["products"] == []
    assert command.update["product_matches"]["bottom-1"]["products"][0]["product_id"] == "102"
    assert command.update["selected_products"]["bottom-1"]["product_id"] == "102"
    assert "product_matches" in command.update["messages"][0].content
    assert command.update["outfit_version"] == 2


@pytest.mark.asyncio
async def test_instruction_only_revision_keeps_selected_item_without_search(outfit):
    revised = outfit.model_copy(deep=True)
    revised.items[0].styling_instruction = "Roll the sleeves"
    product = {
        "product_id": "101", "name": "white-shirt.png", "category": "shirt",
        "color": "white", "similarity": 0.9,
    }
    bottom = {
        "product_id": "102", "name": "black trousers", "category": "pants",
        "color": "black", "similarity": 0.8,
    }
    state = {
        "user_id": "alice", "current_outfit": outfit.model_dump(mode="json"),
        "product_matches": {
            "top-1": {"products": [product]},
            "bottom-1": {"products": [bottom]},
        },
        "selected_products": {"top-1": product},
    }
    search = FakeSearch(None, items=(
        WardrobeItemSummary(
            item_id="101", name="white-shirt.png", category="shirt", color="white"
        ),
        WardrobeItemSummary(
            item_id="102", name="black trousers", category="pants", color="black"
        ),
    ))
    stylist = FakeStylist(revised)
    tools = build_tools(stylist, search)

    command = await tools[1].coroutine(runtime=runtime(state))

    assert search.request is None
    assert command.update["selected_products"]["top-1"]["product_id"] == "101"
    assert command.update["selected_products"]["top-1"]["name"] == "White shirt"
    assert stylist.request.wardrobe_items[0].name == "White shirt"
    assert len(command.update["product_matches"]) == 2


@pytest.mark.asyncio
async def test_new_outfit_searches_wardrobe_and_passes_latest_user_message(outfit):
    product = Product(
        product_id="101", name="Oxford Shirt", category="top", color="white",
        material="cotton", fit="regular", stock=2, similarity=0.9,
    )
    search = FakeSearch(ProductSearchResult(
        outfit_name=outfit.outfit_name,
        matches=[
            OutfitItemMatch(
                outfit_item_id=item.outfit_item_id,
                requested_item=item,
                products=[product] if item.outfit_item_id == "top-1" else [],
            ) for item in outfit.items
        ],
    ))
    stylist = FakeStylist(outfit)
    tools = build_tools(stylist, search)
    command = await tools[1].coroutine(runtime=runtime({
        "user_id": "alice",
        "messages": [
            HumanMessage(content="Gợi ý trang phục"),
            HumanMessage(content="Show me a blue outfit"),
        ],
    }))
    assert search.request.outfit == outfit
    assert search.request.top_k == 5
    assert stylist.request.user_message == "Show me a blue outfit"
    assert len(stylist.request.wardrobe_items) == 2
    assert search.search_inventory.user_id == "alice"
    assert set(command.update["product_matches"]) == {"top-1", "bottom-1"}
    assert command.update["product_matches"]["top-1"]["products"][0]["product_id"] == "101"


@pytest.mark.asyncio
async def test_no_completed_inventory_skips_stylist_and_search(outfit):
    stylist = FakeStylist(outfit)
    search = FakeSearch(None, items=())
    tools = build_tools(stylist, search)

    command = await tools[1].coroutine(runtime=runtime({"user_id": "alice"}))

    assert command.update["current_outfit"] is None
    assert command.update["product_matches"] == {}
    assert "No completed wardrobe items" in command.update["messages"][0].content
    assert stylist.request is None
    assert search.request is None


@pytest.mark.asyncio
async def test_wardrobe_query_lists_owned_completed_items_without_search(outfit):
    search = FakeSearch(None)
    tools = build_tools(FakeStylist(outfit), search)

    command = await tools[6].coroutine(
        runtime=runtime({"user_id": "alice"}), category="shirt", color="white"
    )

    assert search.inventory_user_id == "alice"
    assert search.request is None
    assert '"count": 1' in command.update["messages"][0].content
    assert '"item_id": "101"' in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_inventory_failure_does_not_change_outfit_state(outfit):
    class FailingSearch(FakeSearch):
        async def inventory(self, user_id):
            raise RuntimeError("database unavailable")

    stylist = FakeStylist(outfit)
    tools = build_tools(stylist, FailingSearch(None))

    with pytest.raises(RuntimeError, match="database unavailable"):
        await tools[1].coroutine(runtime=runtime({"user_id": "alice"}))
    assert stylist.request is None


@pytest.mark.asyncio
async def test_search_failure_does_not_commit_partial_outfit(outfit):
    class FailingSearch(FakeSearch):
        async def search(self, request, inventory=None):
            raise RuntimeError("Milvus unavailable")

    state = {"user_id": "alice", "current_outfit": None, "outfit_version": 0}
    tools = build_tools(FakeStylist(outfit), FailingSearch(None))

    with pytest.raises(RuntimeError, match="Milvus unavailable"):
        await tools[1].coroutine(runtime=runtime(state))
    assert state["current_outfit"] is None
    assert state["outfit_version"] == 0


@pytest.mark.asyncio
async def test_product_only_replacement_preserves_outfit(outfit):
    search = FakeSearch(ProductSearchResult(outfit_name=outfit.outfit_name, matches=[]))
    state = {"current_outfit": outfit.model_dump(mode="json"), "product_matches": {}}
    tools = build_tools(FakeStylist(outfit), search)
    command = await tools[2].coroutine(
        outfit_item_ids=["top-1"], color=None, excluded_item_ids=[],
        alternative_to_item_id="101", feedback="show another", top_k=5,
        runtime=runtime(state),
    )
    assert search.request.alternative_to_product_id == "101"
    assert search.request.top_k == 5
    assert search.search_inventory.user_id == ""
    assert search.inventory_calls == 1
    assert "current_outfit" not in command.update
    assert command.update["feedback_history"][0]["type"] == "wardrobe_search"


@pytest.mark.asyncio
async def test_alternative_reuses_verified_matches_without_embedding(outfit):
    selected = Product(
        product_id="101", name="Oxford Shirt", category="shirt", color="white",
        similarity=0.9,
    )
    alternative = Product(
        product_id="103", name="striped shirt", category="shirt", color="blue",
        similarity=0.8,
    )
    search = FakeSearch(None, items=(
        WardrobeItemSummary(item_id="101", name="Oxford Shirt", category="shirt", color="white"),
        WardrobeItemSummary(item_id="103", name="striped shirt", category="shirt", color="blue"),
    ))
    state = {
        "user_id": "alice", "current_outfit": outfit.model_dump(mode="json"),
        "product_matches": {"top-1": {"products": [
            selected.model_dump(), alternative.model_dump(),
        ]}},
        "selected_products": {"top-1": selected.model_dump()},
    }
    tools = build_tools(FakeStylist(outfit), search)

    command = await tools[2].coroutine(
        outfit_item_ids=["top-1"], alternative_to_item_id="101",
        runtime=runtime(state),
    )

    assert search.request is None
    assert "selected_products" not in command.update
    assert state["selected_products"]["top-1"]["product_id"] == "101"
    assert command.update["product_matches"]["top-1"]["products"][0]["product_id"] == "103"
    assert '"product_id": "101"' not in command.update["messages"][0].content


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
async def test_selection_requires_current_users_inventory(outfit):
    product = Product(
        product_id="101", name="Oxford Shirt", category="shirt", color="white",
        similarity=0.9,
    )
    state = {
        "user_id": "alice", "current_outfit": outfit.model_dump(mode="json"),
        "product_matches": {"top-1": {"products": [product.model_dump()]}},
        "selected_products": {},
    }
    available = FakeSearch(None)
    tools = build_tools(FakeStylist(outfit), available)

    command = await tools[4].coroutine(
        outfit_item_id="top-1", item_id="101", runtime=runtime(state)
    )
    assert command.update["selected_products"]["top-1"]["product_id"] == "101"
    assert available.inventory_user_id == "alice"

    unavailable = FakeSearch(None, items=())
    tools = build_tools(FakeStylist(outfit), unavailable)
    with pytest.raises(ValueError, match="no longer available"):
        await tools[4].coroutine(
            outfit_item_id="top-1", item_id="101", runtime=runtime(state)
        )

    duplicate_state = dict(state, selected_products={"bottom-1": product.model_dump()})
    with pytest.raises(ValueError, match="already selected"):
        await build_tools(FakeStylist(outfit), available)[4].coroutine(
            outfit_item_id="top-1", item_id="101", runtime=runtime(duplicate_state)
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
