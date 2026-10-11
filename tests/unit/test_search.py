import pytest

from search.repository import MilvusProductRepository, MilvusSchemaError
from search.schema import Product, ProductConstraints, SearchRequest
from search.service import ProductSearchService


class FakeEmbedding:
    dimension = 3

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRepository:
    metric = "COSINE"

    def __init__(self, products=None, error=None):
        self.products = products or []
        self.error = error
        self.requests = []

    async def search(self, item, vector, constraints, top_k):
        self.requests.append((item, constraints, top_k))
        if self.error:
            raise self.error
        return self.products

    async def validate(self):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_search_preserves_item_association(outfit):
    product = Product(
        product_id="101", name="White Shirt", category="top", color="white",
        price=250_000, stock=2, similarity=0.91,
    )
    repository = FakeRepository([product])
    service = ProductSearchService(repository=repository, embedding=FakeEmbedding())
    result = await service.search(SearchRequest(outfit=outfit, outfit_item_ids=["top-1"]))
    assert result.matches[0].outfit_item_id == "top-1"
    assert result.matches[0].products[0].product_id == "101"


@pytest.mark.asyncio
async def test_search_no_matches_is_truthful(outfit):
    service = ProductSearchService(repository=FakeRepository(), embedding=FakeEmbedding())
    result = await service.search(SearchRequest(outfit=outfit))
    assert all(match.products == [] for match in result.matches)


@pytest.mark.asyncio
async def test_search_failure_propagates(outfit):
    service = ProductSearchService(
        repository=FakeRepository(error=RuntimeError("Milvus unavailable")),
        embedding=FakeEmbedding(),
    )
    with pytest.raises(RuntimeError, match="Milvus unavailable"):
        await service.search(SearchRequest(outfit=outfit))


def test_filter_keeps_hard_constraints(outfit):
    repo = MilvusProductRepository(client=object())
    expr = repo._filter(outfit.items[0], ProductConstraints(
        color="navy", excluded_materials=["leather"], max_price=300_000,
        excluded_product_ids=["bad-id"],
    ))
    assert 'color == "navy"' in expr
    assert 'material != "leather"' in expr
    assert "price <= 300000.0" in expr
    assert 'product_id not in ["bad-id"]' in expr


def test_schema_validation_fails_actionably(monkeypatch):
    class Client:
        def has_collection(self, **_): return True
        def describe_collection(self, **_): return {"fields": [{"name": "wrong"}]}
    repo = MilvusProductRepository(client=Client())
    with pytest.raises(MilvusSchemaError, match="missing fields"):
        repo._validate_sync()
