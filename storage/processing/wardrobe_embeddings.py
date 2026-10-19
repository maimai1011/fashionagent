"""Store garment embeddings in the image features Milvus collection."""

import os
from uuid import UUID

from storage.setup.milvus import connect_milvus, create_collection, disconnect_milvus


def insert_wardrobe_embedding(item_id: UUID, embedding: list[float]) -> str:
    """Insert one CLIP embedding and return a reference for PostgreSQL."""
    if not embedding:
        raise ValueError("garment embedding cannot be empty")

    connect_milvus()
    try:
        collection = create_collection()
        vector_field = next(
            (field for field in collection.schema.fields if field.name == "embedding"),
            None,
        )
        if vector_field is None:
            raise ValueError("Milvus image collection has no embedding field")
        dimension = int(vector_field.params["dim"])

        if len(embedding) != dimension:
            raise ValueError(
                f"CLIP embedding has {len(embedding)} values, "
                f"but Milvus collection expects {dimension}"
            )

        result = collection.insert(
            [{"image_name": str(item_id), "embedding": embedding}]
        )
        if len(result.primary_keys) != 1:
            raise RuntimeError("Milvus did not return an embedding ID")
        collection_name = os.getenv("MILVUS_COLLECTION", "image_features")
        return f"{collection_name}:{result.primary_keys[0]}"
    finally:
        disconnect_milvus()
