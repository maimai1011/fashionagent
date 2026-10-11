"""Metric-aware candidate scoring and deterministic reranking."""

from search.schema import MetricType, Product


def normalized_similarity(distance: float, metric: MetricType) -> float:
    if metric == "L2":
        return 1.0 / (1.0 + max(distance, 0.0))
    return float(distance)


def rerank(products: list[Product], metric: MetricType) -> list[Product]:
    del metric  # all products already carry normalized similarity
    return sorted(products, key=lambda item: (item.similarity, bool(item.stock)), reverse=True)
