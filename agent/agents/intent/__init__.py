"""Dynamic coordinator agent."""

import logging

logger = logging.getLogger(__name__)

__all__ = ["FashionState", "create_intent_agent"]


def __getattr__(name: str) -> object:
    logger.info("function_start function=agents.intent.__getattr__")
    if name == "create_intent_agent":
        from .agent import create_intent_agent

        return create_intent_agent
    if name == "FashionState":
        from .state import FashionState

        return FashionState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
