"""Checkpointed state for the coordinator agent."""

import operator
from typing import Annotated

from langchain.agents.middleware import AgentState


class FashionState(AgentState):
    user_id: str
    current_outfit: dict[str, object] | None
    product_matches: dict[str, dict[str, object]]
    selected_products: dict[str, dict[str, object]]
    preferences: dict[str, object]
    feedback_history: list[dict[str, object]]
    outfit_version: int
    outfit_history: list[dict[str, object]]
    tool_call_count: Annotated[int, operator.add]
