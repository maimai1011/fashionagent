"""Wardrobe and legacy catalog search helpers."""

__all__ = ["ProductSearchResult", "ProductSearchService", "SearchRequest"]


def __getattr__(name: str) -> object:
    if name in {"ProductSearchResult", "SearchRequest"}:
        from .schema import ProductSearchResult, SearchRequest

        return {"ProductSearchResult": ProductSearchResult, "SearchRequest": SearchRequest}[name]
    if name == "ProductSearchService":
        from .service import ProductSearchService

        return ProductSearchService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
