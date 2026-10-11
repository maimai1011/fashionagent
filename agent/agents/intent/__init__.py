"""Dynamic coordinator agent."""

from typing import Any

__all__ = ["FashionState", "create_intent_agent"]


def __getattr__(name: str) -> Any:
    if name == "create_intent_agent":
        from .agent import create_intent_agent

        return create_intent_agent
    if name == "FashionState":
        from .state import FashionState

        return FashionState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
