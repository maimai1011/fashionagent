"""Checkpointed state for the coordinator agent."""

import operator
from typing import Annotated, Any

from langchain.agents.middleware import AgentState


class FashionState(AgentState):
    user_id: str
    current_outfit: dict[str, Any] | None
    product_matches: dict[str, dict[str, Any]]
    selected_products: dict[str, dict[str, Any]]
    preferences: dict[str, Any]
    feedback_history: list[dict[str, Any]]
    outfit_version: int
    outfit_history: list[dict[str, Any]]
    tool_call_count: Annotated[int, operator.add]
