"""Milvus connection and collection setup."""

import logging
import os

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

logger = logging.getLogger(__name__)


def _embedding_dimension() -> int:
    logger.info("function_start function=setup.milvus._embedding_dimension")
    raw_value = os.getenv("MILVUS_EMBEDDING_DIM", "512")
    try:
        dimension = int(raw_value)
    except ValueError as exc:
        raise ValueError("MILVUS_EMBEDDING_DIM must be an integer") from exc
    if dimension <= 0:
        raise ValueError("MILVUS_EMBEDDING_DIM must be greater than zero")
    return dimension


def connect_milvus() -> None:
    logger.info("function_start function=setup.milvus.connect_milvus")
    logger.info("Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.environ["MILVUS_PORT"],
    )
    logger.info("Connected to Milvus")


def create_collection() -> Collection:
    """Create the configured image embedding collection if necessary."""
    logger.info("function_start function=setup.milvus.create_collection")
    collection_name = os.getenv("MILVUS_COLLECTION", "image_features")
    embedding_dim = _embedding_dimension()

    # Check exists
    if utility.has_collection(collection_name):
        logger.info("Milvus collection '%s' already exists", collection_name)
        collection = Collection(collection_name)
        _ensure_embedding_index(collection)
        return collection

    logger.info("Creating collection...")

    fields = [
        FieldSchema(
            name='id',
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="image_name",
            dtype=DataType.VARCHAR,
            max_length=255
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim
        )
    ]

    schema = CollectionSchema(fields, description="image feature")

    collection = Collection(
        name=collection_name,
        schema=schema
    )

    _ensure_embedding_index(collection)

    logger.info("Created Milvus collection '%s'", collection_name)
    return collection


def _ensure_embedding_index(collection: Collection) -> None:
    """Make CLIP embeddings searchable without altering existing vectors."""
    for index in collection.indexes:
        if index.field_name == "embedding":
            metric = str(index.params.get("metric_type", "")).upper()
            if metric != "COSINE":
                raise ValueError(
                    f"Milvus image embedding index must use COSINE, not {metric or 'unknown'}"
                )
            return

    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
    )
    logger.info("Created COSINE FLAT index for '%s'", collection.name)


def disconnect_milvus() -> None:
    """Close the default Milvus connection."""
    logger.info("function_start function=setup.milvus.disconnect_milvus")
    connections.disconnect(alias="default")
