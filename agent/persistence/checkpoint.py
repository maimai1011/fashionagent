"""Development and production LangGraph checkpointer lifecycle."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fashion_agent.config import Settings, settings


@asynccontextmanager
async def checkpointer_context(config: Settings = settings) -> AsyncIterator[Any]:
    if config.checkpoint_backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        yield InMemorySaver()
        return
    if config.checkpoint_backend != "postgres":
        raise ValueError("CHECKPOINT_BACKEND must be 'memory' or 'postgres'")
    if not config.checkpoint_database_url:
        raise RuntimeError("CHECKPOINT_DATABASE_URL is required for postgres checkpoints")
    if os.getenv("LANGGRAPH_STRICT_MSGPACK", "").lower() != "true":
        raise RuntimeError(
            "LANGGRAPH_STRICT_MSGPACK=true is required for production checkpoints"
        )
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(config.checkpoint_database_url) as saver:
        await saver.setup()
        yield saver
