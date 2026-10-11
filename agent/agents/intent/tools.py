"""State-aware tools dynamically selected by the Intent Agent."""

import json
import logging
from copy import deepcopy
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from agent.agents.stylist.agent import StylistAgent
from agent.agents.stylist.schema import OutfitPlan, StylistRequest
from search.schema import ProductConstraints, SearchRequest
from search.service import ProductSearchService


logger = logging.getLogger(__name__)


def _tool_message(runtime: ToolRuntime, payload: dict[str, Any]) -> list[ToolMessage]:
    logger.info("function_start function=_tool_message")
    return [ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id or "unknown-tool-call",
    )]


def _snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    logger.info("function_start function=_snapshot")
    if not state.get("current_outfit"):
        return None
    return deepcopy({
        "version": state.get("outfit_version", 0),
        "outfit": state["current_outfit"],
        "product_matches": state.get("product_matches", {}),
        "selected_products": state.get("selected_products", {}),
    })


def _unchanged_item_ids(old: OutfitPlan | None, new: OutfitPlan) -> set[str]:
    logger.info("function_start function=_unchanged_item_ids")
    if old is None:
        return set()
    old_items = {item.outfit_item_id: item for item in old.items}
    return {
        item.outfit_item_id for item in new.items
        if old_items.get(item.outfit_item_id) == item
    }


def build_tools(stylist: StylistAgent, product_search: ProductSearchService) -> list[Any]:
    logger.info("function_start function=build_tools")
    @tool
    async def ask_user(question: str, runtime: ToolRuntime) -> Command:
        """Ask one essential clarification question and pause until the user answers."""
        logger.info("function_start function=ask_user")
        answer = interrupt({"type": "clarification", "question": question})
        count = int(runtime.state.get("tool_call_count", 0)) + 1
        payload = {"clarification_answer": str(answer)}
        logger.info("tool_complete tool=ask_user tool_call_count=%d", count)
        return Command(update={
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    @tool
    async def consult_stylist(
        runtime: ToolRuntime,
        preferences: dict[str, Any] | None = None,
        feedback: str = "",
        keep_categories: list[str] | None = None,
        keep_outfit_item_ids: list[str] | None = None,
    ) -> Command:
        """Create or revise an outfit through the independent Stylist Agent."""
        logger.info("function_start function=consult_stylist")
        state = runtime.state
        current = (
            OutfitPlan.model_validate(state["current_outfit"])
            if state.get("current_outfit") else None
        )
        merged_preferences = deepcopy(state.get("preferences", {}))
        merged_preferences.update(preferences or {})
        request = StylistRequest(
            preferences=merged_preferences,
            current_outfit=current,
            feedback=feedback,
            keep_categories=keep_categories or [],
            keep_outfit_item_ids=keep_outfit_item_ids or [],
        )
        result = await stylist.run(request)
        unchanged = _unchanged_item_ids(current, result)
        old_matches = state.get("product_matches", {})
        old_selected = state.get("selected_products", {})
        matches = {key: deepcopy(value) for key, value in old_matches.items() if key in unchanged}
        selected = {key: deepcopy(value) for key, value in old_selected.items() if key in unchanged}
        history = deepcopy(state.get("outfit_history", []))
        if snapshot := _snapshot(state):
            history.append(snapshot)
        version = int(state.get("outfit_version", 0)) + 1
        feedback_history = deepcopy(state.get("feedback_history", []))
        if feedback:
            feedback_history.append({"feedback": feedback, "outfit_version": version})
        payload = {"outfit": result.model_dump(mode="json"), "outfit_version": version}
        count = int(state.get("tool_call_count", 0)) + 1
        logger.info(
            "tool_complete tool=consult_stylist outfit_version=%d tool_call_count=%d "
            "feedback_revision_count=%d",
            version,
            count,
            len(feedback_history),
        )
        return Command(update={
            "current_outfit": payload["outfit"],
            "outfit_version": version,
            "outfit_history": history,
            "preferences": merged_preferences,
            "feedback_history": feedback_history,
            "product_matches": matches,
            "selected_products": selected,
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    @tool
    async def search_products(
        runtime: ToolRuntime,
        outfit_item_ids: list[str] | None = None,
        color: str | None = None,
        excluded_materials: list[str] | None = None,
        max_price: float | None = None,
        excluded_product_ids: list[str] | None = None,
        alternative_to_product_id: str | None = None,
        feedback: str = "",
        top_k: int = 5,
    ) -> Command:
        """Search real Milvus products for all or selected current outfit items."""
        logger.info("function_start function=search_products")
        if not runtime.state.get("current_outfit"):
            raise ValueError("No current outfit exists; consult the stylist first")
        outfit = OutfitPlan.model_validate(runtime.state["current_outfit"])
        request = SearchRequest(
            outfit=outfit,
            outfit_item_ids=outfit_item_ids or [],
            top_k=top_k,
            alternative_to_product_id=alternative_to_product_id,
            constraints=ProductConstraints(
                color=color,
                excluded_materials=excluded_materials or [],
                max_price=max_price,
                excluded_product_ids=excluded_product_ids or [],
            ),
        )
        result = await product_search.search(request)
        matches = deepcopy(runtime.state.get("product_matches", {}))
        for match in result.matches:
            matches[match.outfit_item_id] = match.model_dump(mode="json")
        payload = result.model_dump(mode="json")
        feedback_history = deepcopy(runtime.state.get("feedback_history", []))
        if feedback:
            feedback_history.append({
                "feedback": feedback,
                "outfit_version": runtime.state.get("outfit_version", 0),
                "type": "product_search",
            })
        count = int(runtime.state.get("tool_call_count", 0)) + 1
        logger.info(
            "tool_complete tool=search_products item_count=%d result_count=%d "
            "tool_call_count=%d",
            len(result.matches),
            sum(len(match.products) for match in result.matches),
            count,
        )
        return Command(update={
            "product_matches": matches,
            "feedback_history": feedback_history,
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    @tool
    async def restore_previous_outfit(runtime: ToolRuntime) -> Command:
        """Restore the exact previous outfit and its saved product state."""
        logger.info("function_start function=restore_previous_outfit")
        history = deepcopy(runtime.state.get("outfit_history", []))
        if not history:
            raise ValueError("There is no previous outfit version to restore")
        previous = history.pop()
        redo_history = history
        payload = {
            "outfit": previous["outfit"],
            "outfit_version": previous["version"],
            "restored": True,
        }
        logger.info(
            "tool_complete tool=restore_previous_outfit outfit_version=%d",
            previous["version"],
        )
        count = int(runtime.state.get("tool_call_count", 0)) + 1
        return Command(update={
            "current_outfit": deepcopy(previous["outfit"]),
            "product_matches": deepcopy(previous["product_matches"]),
            "selected_products": deepcopy(previous["selected_products"]),
            "outfit_version": previous["version"],
            "outfit_history": redo_history,
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    @tool
    async def select_product(
        outfit_item_id: str,
        product_id: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Select a verified product already returned for an outfit item."""
        logger.info("function_start function=select_product")
        match = runtime.state.get("product_matches", {}).get(outfit_item_id)
        if not match:
            raise ValueError(f"No product results exist for outfit item '{outfit_item_id}'")
        product = next(
            (
                item for item in match.get("products", [])
                if str(item.get("product_id")) == product_id
            ),
            None,
        )
        if product is None:
            raise ValueError("The requested product ID is not present in verified search results")
        selected = deepcopy(runtime.state.get("selected_products", {}))
        selected[outfit_item_id] = deepcopy(product)
        payload = {"outfit_item_id": outfit_item_id, "selected_product": product}
        logger.info(
            "tool_complete tool=select_product outfit_item_id=%s", outfit_item_id
        )
        count = int(runtime.state.get("tool_call_count", 0)) + 1
        return Command(update={
            "selected_products": selected,
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    return [
        ask_user,
        consult_stylist,
        search_products,
        restore_previous_outfit,
        select_product,
    ]
