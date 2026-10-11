import os

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="set RUN_INTEGRATION_TESTS=true with live vLLM and Milvus",
)
@pytest.mark.asyncio
async def test_live_services_are_explicitly_opt_in():
    from agents.stylist import StylistAgent, StylistRequest
    from search.schema import SearchRequest
    from search.service import ProductSearchService

    search = ProductSearchService()
    await search.validate()
    result = await StylistAgent().run(StylistRequest(preferences={"occasion": "work"}))
    assert result.items
    matches = await search.search(SearchRequest(outfit=result, top_k=1))
    assert len(matches.matches) == len(result.items)
    search.close()
