"""Evaluate saved wardrobe retrieval results against labeled JSONL queries.

Each line contains requested_category, requested_name, relevant_item_ids,
and results: [{item_id, category, name}, ...] in returned rank order.
"""

import argparse
import json
from pathlib import Path

from agent.search.categories import category_compatible


def evaluate(records: list[dict[str, object]], k: int = 5) -> dict[str, float | int]:
    """Return mean Precision@K and the fraction of returned category mismatches."""
    if k < 1:
        raise ValueError("k must be positive")
    precision_sum = 0.0
    labeled_queries = 0
    returned = 0
    mismatched = 0
    for record in records:
        relevant = set(record["relevant_item_ids"])
        results = record["results"][:k]
        if relevant:
            precision_sum += sum(
                result["item_id"] in relevant for result in results
            ) / k
            labeled_queries += 1
        for result in results:
            returned += 1
            if not category_compatible(
                str(record["requested_category"]), str(record["requested_name"]),
                str(result["category"]), str(result["name"]),
            ):
                mismatched += 1
    return {
        "queries": len(records),
        "queries_with_relevant_items": labeled_queries,
        "k": k,
        "precision_at_k": precision_sum / labeled_queries if labeled_queries else 0.0,
        "category_mismatch_rate": mismatched / returned if returned else 0.0,
        "returned_items": returned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Labeled JSONL retrieval results")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(records, args.k), indent=2))


if __name__ == "__main__":
    main()
