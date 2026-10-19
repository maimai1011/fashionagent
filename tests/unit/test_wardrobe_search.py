"""Wardrobe search must use uploaded vectors and the current user's records."""

import os
import unittest
from unittest.mock import patch
from uuid import UUID

from agent.agents.stylist.schema import OutfitItem, OutfitPlan, WardrobeItemSummary
from agent.search.categories import (
    category_compatible,
    normalize_category,
    normalize_color,
)
from agent.search.schema import ProductConstraints, SearchRequest
from agent.search.wardrobe import (
    MilvusWardrobeRepository,
    WardrobeInventory,
    WardrobeSearchService,
)

ITEM = OutfitItem(
    outfit_item_id="top-1",
    category="top",
    item_name="shirt",
    color="blue",
    material="cotton",
    fit="regular",
    styling_instruction="Wear with jeans",
)
OUTFIT = OutfitPlan(
    outfit_name="Casual",
    occasion="everyday",
    style="casual",
    color_palette=["blue"],
    items=[ITEM],
    reasoning="A simple outfit",
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.params = None
        self.query = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, query, params):
        if query.count("%s") != len(params):
            raise TypeError("SQL placeholder count does not match parameters")
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


class FakeMilvus:
    def __init__(self, item_id, other_id, hits=None):
        self.item_id = item_id
        self.other_id = other_id
        self.hits = hits
        self.search_kwargs = None
        self.loaded_collection = None

    def load_collection(self, **kwargs):
        self.loaded_collection = kwargs["collection_name"]

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [self.hits if self.hits is not None else [
            {"entity": {"image_name": self.item_id}, "distance": 0.9},
            {"entity": {"image_name": self.other_id}, "distance": 0.8},
        ]]


class WardrobeRepositoryTests(unittest.TestCase):
    def test_category_aliases_use_name_and_reject_cross_type_matches(self):
        self.assertEqual(normalize_category("accessory", "gold watch.png"), "watch")
        self.assertEqual(normalize_category("top", "white Oxford shirt"), "shirt")
        self.assertTrue(category_compatible("shoes", "sneakers", "shoe", "black boots"))
        self.assertFalse(category_compatible("shoes", "sneakers", "skirt", "black skirt"))
        self.assertFalse(category_compatible("top", "blouse", "watch", "gold watch"))
        self.assertEqual(normalize_category("áo", "áo sơ mi trắng"), "shirt")
        self.assertEqual(normalize_category("váy", "chân váy đen"), "skirt")
        self.assertEqual(normalize_color("đỏ"), "red")
        self.assertTrue(category_compatible("giày", "giày thể thao", "shoes", "sneakers"))

    def test_count_includes_only_completed_items_for_one_user_without_milvus(self):
        connection = FakeConnection([(2,)])
        milvus = FakeMilvus("unused", "other")
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                count = repository._count_items_sync("alice")
        finally:
            repository.close()

        self.assertEqual(count, 2)
        self.assertIn(
            "WHERE user_id = %s AND status = 'completed'",
            connection.cursor_instance.query,
        )
        self.assertEqual(connection.cursor_instance.params, ("alice",))
        self.assertTrue(connection.closed)
        self.assertIsNone(milvus.search_kwargs)

    def test_search_joins_only_current_users_completed_uploads(self):
        item_id = str(UUID("00000000-0000-0000-0000-000000000001"))
        other_id = str(UUID("00000000-0000-0000-0000-000000000002"))
        connection = FakeConnection([(UUID(item_id), "blue-shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(item_id, other_id)
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection), patch.dict(
                os.environ, {"API_BASE_URL": "http://localhost:8002"}
            ):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()

        self.assertEqual(connection.cursor_instance.params, ("alice",))
        self.assertTrue(connection.closed)
        self.assertEqual(milvus.search_kwargs["collection_name"], "image_features")
        self.assertEqual(milvus.loaded_collection, "image_features")
        self.assertIn(item_id, milvus.search_kwargs["filter"])
        self.assertNotIn(other_id, milvus.search_kwargs["filter"])
        self.assertEqual([product.product_id for product in results], [item_id])
        self.assertEqual(results[0].name, "Blue shirt")
        self.assertIsNone(results[0].stock)
        self.assertEqual(results[0].image_url, f"http://localhost:8002/v1/wardrobe/{item_id}/image")

    def test_no_completed_uploads_skips_milvus(self):
        connection = FakeConnection([])
        milvus = FakeMilvus("unused", "other")
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual(results, [])
        self.assertIsNone(milvus.search_kwargs)

    def test_category_filter_excludes_wrong_garments_before_milvus(self):
        shoe_id = str(UUID("00000000-0000-0000-0000-000000000011"))
        skirt_id = str(UUID("00000000-0000-0000-0000-000000000012"))
        connection = FakeConnection([
            (UUID(shoe_id), "running sneakers.png", "blouse", "black"),
            (UUID(skirt_id), "black skirt.png", "skirt", "black"),
        ])
        milvus = FakeMilvus(shoe_id, skirt_id)
        repository = MilvusWardrobeRepository(client=milvus)
        shoes = ITEM.model_copy(update={"category": "shoes", "item_name": "sneakers"})
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    shoes, [0.01] * 512, ProductConstraints(color="black"), 5, "alice"
                )
        finally:
            repository.close()

        self.assertEqual([item.product_id for item in results], [shoe_id])
        self.assertEqual(results[0].category, "shoes")
        self.assertIn(shoe_id, milvus.search_kwargs["filter"])
        self.assertNotIn(skirt_id, milvus.search_kwargs["filter"])

    def test_low_similarity_returns_no_match(self):
        shirt_id = str(UUID("00000000-0000-0000-0000-000000000013"))
        connection = FakeConnection([(UUID(shirt_id), "blue shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(shirt_id, "other", hits=[
            {"entity": {"image_name": shirt_id}, "distance": 0.1},
        ])
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual(results, [])

    def test_duplicate_hits_keep_best_score_once(self):
        shirt_id = str(UUID("00000000-0000-0000-0000-000000000015"))
        connection = FakeConnection([(UUID(shirt_id), "blue shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(shirt_id, "other", hits=[
            {"entity": {"image_name": shirt_id}, "distance": 0.4},
            {"entity": {"image_name": shirt_id}, "distance": 0.9},
        ])
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].similarity, 0.9)

    def test_reranking_uses_color_and_stable_ids(self):
        red_id = str(UUID("00000000-0000-0000-0000-000000000016"))
        blue_id = str(UUID("00000000-0000-0000-0000-000000000017"))
        connection = FakeConnection([
            (UUID(red_id), "red shirt.png", "shirt", "red"),
            (UUID(blue_id), "blue shirt.png", "shirt", "blue"),
        ])
        milvus = FakeMilvus(red_id, blue_id, hits=[
            {"entity": {"image_name": red_id}, "distance": 0.82},
            {"entity": {"image_name": blue_id}, "distance": 0.81},
        ])
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual([item.product_id for item in results], [blue_id, red_id])

    def test_similarity_threshold_is_configurable(self):
        shirt_id = str(UUID("00000000-0000-0000-0000-000000000018"))
        connection = FakeConnection([(UUID(shirt_id), "blue shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(shirt_id, "other", hits=[
            {"entity": {"image_name": shirt_id}, "distance": 0.8},
        ])
        with patch.dict(os.environ, {"WARDROBE_MIN_SIMILARITY": "0.85"}):
            repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual(results, [])

    def test_explicit_exclusions_and_color_filter_before_vector_search(self):
        shirt_id = str(UUID("00000000-0000-0000-0000-000000000014"))
        connection = FakeConnection([(UUID(shirt_id), "blue shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(shirt_id, "other")
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                excluded = repository._search_sync(
                    ITEM, [0.01] * 512,
                    ProductConstraints(excluded_product_ids=[shirt_id]), 5, "alice"
                )
                wrong_color = repository._search_sync(
                    ITEM, [0.01] * 512, ProductConstraints(color="red"), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual(excluded, [])
        self.assertEqual(wrong_color, [])
        self.assertIsNone(milvus.search_kwargs)

    def test_vietnamese_color_constraint_matches_stored_english_color(self):
        shirt_id = str(UUID("00000000-0000-0000-0000-000000000019"))
        connection = FakeConnection([(UUID(shirt_id), "blue shirt.png", "shirt", "blue")])
        milvus = FakeMilvus(shirt_id, "other")
        repository = MilvusWardrobeRepository(client=milvus)
        try:
            with patch.object(MilvusWardrobeRepository, "_get_connection", return_value=connection):
                results = repository._search_sync(
                    ITEM, [0.01] * 512,
                    ProductConstraints(color="xanh dương"), 5, "alice"
                )
        finally:
            repository.close()
        self.assertEqual([item.product_id for item in results], [shirt_id])


class FakeEmbedding:
    async def embed(self, texts):
        self.texts = texts
        return [[0.01] * 512 for _ in texts]


class FakeRepository:
    metric = "COSINE"

    async def inventory(self, user_id):
        self.user_id = user_id
        self.inventory_calls = getattr(self, "inventory_calls", 0) + 1
        return WardrobeInventory(user_id, (
            WardrobeItemSummary(
                item_id="shirt-1", name="blue-shirt.png", category="shirt", color="blue"
            ),
        ))

    eligible_items = staticmethod(MilvusWardrobeRepository.eligible_items)

    async def search_prepared(self, item, vector, top_k, eligible):
        self.item = item
        self.vector = vector
        self.eligible = eligible
        return []

    async def count_items(self, user_id):
        self.user_id = user_id
        return 2


class WardrobeServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_search_request_caps_matches_per_item_at_five(self):
        for top_k in (4, 6, 100):
            with self.assertRaises(ValueError):
                SearchRequest(outfit=OUTFIT, user_id="alice", top_k=top_k)

    async def test_count_does_not_require_an_outfit(self):
        repository = FakeRepository()
        service = WardrobeSearchService(repository=repository, embedding=FakeEmbedding())
        self.assertEqual(
            await service.count_items("alice"),
            2,
        )
        self.assertEqual(repository.user_id, "alice")

    async def test_search_passes_user_id_and_clip_query(self):
        repository = FakeRepository()
        embedding = FakeEmbedding()
        service = WardrobeSearchService(repository=repository, embedding=embedding)
        result = await service.search(SearchRequest(outfit=OUTFIT, user_id="alice"))
        self.assertEqual(repository.user_id, "alice")
        self.assertEqual(embedding.texts, ["a photo of a blue shirt, shirt clothing"])
        self.assertEqual(result.matches[0].products, [])
        self.assertEqual(repository.inventory_calls, 1)

    async def test_missing_category_skips_embedding_and_milvus(self):
        repository = FakeRepository()
        embedding = FakeEmbedding()
        service = WardrobeSearchService(repository=repository, embedding=embedding)
        shoes = ITEM.model_copy(update={"category": "shoes", "item_name": "sneakers"})
        outfit = OUTFIT.model_copy(update={"items": [shoes]})

        result = await service.search(SearchRequest(outfit=outfit, user_id="alice"))

        self.assertEqual(result.matches[0].products, [])
        self.assertFalse(hasattr(embedding, "texts"))
        self.assertFalse(hasattr(repository, "item"))

    async def test_rejects_another_users_inventory(self):
        service = WardrobeSearchService(repository=FakeRepository(), embedding=FakeEmbedding())
        inventory = WardrobeInventory("bob", ())
        with self.assertRaisesRegex(ValueError, "different user"):
            await service.search(SearchRequest(outfit=OUTFIT, user_id="alice"), inventory)

    async def test_identical_clip_queries_are_embedded_once(self):
        repository = FakeRepository()
        embedding = FakeEmbedding()
        service = WardrobeSearchService(repository=repository, embedding=embedding)
        another = ITEM.model_copy(update={"outfit_item_id": "top-2"})
        outfit = OUTFIT.model_copy(update={"items": [ITEM, another]})

        result = await service.search(SearchRequest(outfit=outfit, user_id="alice"))

        self.assertEqual(len(result.matches), 2)
        self.assertEqual(len(embedding.texts), 1)
        self.assertEqual(repository.inventory_calls, 1)

    async def test_vietnamese_outfit_uses_english_clip_query(self):
        repository = FakeRepository()
        embedding = FakeEmbedding()
        service = WardrobeSearchService(repository=repository, embedding=embedding)
        item = ITEM.model_copy(update={
            "category": "áo", "item_name": "áo sơ mi", "color": "xanh dương"
        })
        outfit = OUTFIT.model_copy(update={"items": [item]})

        await service.search(SearchRequest(outfit=outfit, user_id="alice"))

        self.assertEqual(embedding.texts, ["a photo of a blue shirt, shirt clothing"])

    async def test_search_requires_user_id(self):
        service = WardrobeSearchService(repository=FakeRepository(), embedding=FakeEmbedding())
        with self.assertRaisesRegex(ValueError, "user_id is required"):
            await service.search(SearchRequest(outfit=OUTFIT))


if __name__ == "__main__":
    unittest.main()
