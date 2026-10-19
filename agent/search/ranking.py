"""Metric-aware candidate scoring and deterministic reranking."""

import logging

from agent.search.schema import MetricType, Product

logger = logging.getLogger(__name__)


def normalized_similarity(distance: float, metric: MetricType) -> float:
    logger.info("function_start function=normalized_similarity metric=%s", metric)
    if metric == "L2":
        return 1.0 / (1.0 + max(distance, 0.0))
    return float(distance)


def rerank(products: list[Product], metric: MetricType) -> list[Product]:
    logger.info("function_start function=rerank count=%d metric=%s", len(products), metric)
    del metric  # all products already carry normalized similarity
    return sorted(products, key=lambda item: (item.similarity, bool(item.stock)), reverse=True)
