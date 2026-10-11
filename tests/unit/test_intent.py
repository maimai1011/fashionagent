from agents.intent.prompt import INTENT_PROMPT


def test_prompt_delegates_styling_and_forbids_fabrication():
    assert "not a stylist" in INTENT_PROMPT
    assert "Never invent product IDs" in INTENT_PROMPT
    assert "search_products" in INTENT_PROMPT
    assert "restore_previous_outfit" in INTENT_PROMPT
    assert "select_product" in INTENT_PROMPT
