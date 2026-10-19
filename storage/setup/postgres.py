"""PostgreSQL connection setup."""

import logging
import os

import psycopg2
from psycopg2.extras import register_uuid

logger = logging.getLogger(__name__)

# Wardrobe item IDs are uuid.UUID objects in the API and Celery worker.
# Register their PostgreSQL adapter and result typecaster in each process.
register_uuid()


def get_connection():
    """Create a psycopg2 connection from environment configuration."""

    logger.info("Connecting to Postgres")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.environ["POSTGRES_PORT"],
    )


def create_tables(query: str) -> None:
    """Execute a CREATE TABLE statement in a managed transaction."""
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    connection = get_connection()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(query)
        logger.info("Postgres table creation completed")
    except Exception:
        logger.exception("Failed to create Postgres tables")
        raise
    finally:
        connection.close()
