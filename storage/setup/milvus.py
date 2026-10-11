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
    raw_value = os.getenv("MILVUS_EMBEDDING_DIM", "512")
    try:
        dimension = int(raw_value)
    except ValueError as exc:
        raise ValueError("MILVUS_EMBEDDING_DIM must be an integer") from exc
    if dimension <= 0:
        raise ValueError("MILVUS_EMBEDDING_DIM must be greater than zero")
    return dimension


def connect_milvus() -> None:
    logger.info("Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    logger.info("Connected to Milvus")


def create_collection() -> Collection:
    """Create the configured image embedding collection if necessary."""
    collection_name = os.getenv("MILVUS_COLLECTION", "image_features")
    embedding_dim = _embedding_dimension()

    # Check exists
    if utility.has_collection(collection_name):
        logger.info("Milvus collection '%s' already exists", collection_name)
        return Collection(collection_name)

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

    logger.info("Created Milvus collection '%s'", collection_name)
    return collection


def disconnect_milvus() -> None:
    """Close the default Milvus connection."""
    connections.disconnect(alias="default")
