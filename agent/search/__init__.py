"""Milvus-backed catalog search."""

from .schema import ProductSearchResult, SearchRequest
from .service import ProductSearchService

__all__ = ["ProductSearchResult", "ProductSearchService", "SearchRequest"]
