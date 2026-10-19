import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="set RUN_INTEGRATION_TESTS=true with live vLLM and Milvus",
)
@pytest.mark.asyncio
async def test_live_services_are_explicitly_opt_in():
    from agent.agents.stylist import StylistAgent, StylistRequest
    from agent.search.schema import SearchRequest
    from agent.search.wardrobe import WardrobeSearchService

    search = WardrobeSearchService()
    try:
        await search.validate()
        user_id = os.getenv("INTEGRATION_USER_ID", "test-user")
        inventory = await search.inventory(user_id)
        if not inventory.items:
            pytest.skip("INTEGRATION_USER_ID has no completed wardrobe uploads")
        result = await StylistAgent().run(StylistRequest(
            preferences={"occasion": "work"},
            user_message="Suggest a work outfit from my wardrobe",
            wardrobe_items=list(inventory.items),
        ))
        assert result.items
        matches = await search.search(SearchRequest(
            outfit=result,
            user_id=user_id,
        ), inventory=inventory)
        assert len(matches.matches) == len(result.items)
        assert all(len(match.products) <= 5 for match in matches.matches)
    finally:
        search.close()
