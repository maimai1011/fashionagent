"""Checks for offline wardrobe retrieval metrics."""

from agent.search.evaluate import evaluate


def test_precision_and_category_mismatch_rate():
    records = [
        {
            "requested_category": "shoes",
            "requested_name": "sneakers",
            "relevant_item_ids": ["shoe-1"],
            "results": [
                {"item_id": "shoe-1", "category": "footwear", "name": "white sneakers"},
                {"item_id": "skirt-1", "category": "skirt", "name": "black skirt"},
            ],
        },
        {
            "requested_category": "top",
            "requested_name": "blouse",
            "relevant_item_ids": [],
            "results": [],
        },
    ]

    metrics = evaluate(records, k=2)

    assert metrics["precision_at_k"] == 0.5
    assert metrics["category_mismatch_rate"] == 0.5
    assert metrics["queries"] == 2
    assert metrics["queries_with_relevant_items"] == 1
