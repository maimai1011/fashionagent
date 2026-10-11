"""PostgreSQL connection setup."""

import os
import logging
from collections.abc import Iterable
from typing import Any

import psycopg2


logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    logger.info("function_start function=setup.postgres._required_env")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def get_connection() -> Any:
    """Create a psycopg2 connection from environment configuration."""

    logger.info("function_start function=setup.postgres.get_connection")
    logger.info("Connecting to Postgres")
    return psycopg2.connect(
        host=_required_env("POSTGRES_HOST"),
        database=_required_env("POSTGRES_DB"),
        user=_required_env("POSTGRES_USER"),
        password=_required_env("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )


def create_tables(query: str) -> None:
    """Execute a CREATE TABLE statement in a managed transaction."""
    logger.info("function_start function=setup.postgres.create_tables")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
        logger.info("Postgres table creation completed")
    except Exception:
        logger.exception("Failed to create Postgres tables")
        raise
    finally:
        connection.close()


def initialize_postgres(statements: Iterable[str]) -> None:
    """Create all configured tables in one transaction."""
    logger.info("function_start function=setup.postgres.initialize_postgres")
    queries = tuple(statements)
    if not queries:
        raise ValueError("at least one CREATE TABLE statement is required")

    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                for query in queries:
                    if not query or not query.strip():
                        raise ValueError("table statements cannot be empty")
                    cursor.execute(query)
        logger.info("Postgres initialization completed")
    except Exception:
        logger.exception("Failed to initialize Postgres")
        raise
    finally:
        connection.close()
