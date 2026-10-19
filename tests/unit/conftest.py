import pytest

from agent.agents.stylist.schema import OutfitItem, OutfitPlan


@pytest.fixture
def outfit() -> OutfitPlan:
    return OutfitPlan(
        outfit_name="Office Look",
        occasion="work",
        style="elegant",
        color_palette=["white", "black"],
        items=[
            OutfitItem(
                outfit_item_id="top-1", category="top", item_name="Oxford Shirt",
                color="white", material="cotton", fit="regular",
                styling_instruction="Tuck into trousers",
            ),
            OutfitItem(
                outfit_item_id="bottom-1", category="bottom", item_name="Trousers",
                color="black", material="wool", fit="straight",
                styling_instruction="Wear at natural waist",
            ),
        ],
        styling_tips=["Keep accessories minimal"],
        reasoning="Balanced office proportions.",
    )
