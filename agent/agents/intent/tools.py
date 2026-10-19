"""State-aware tools dynamically selected by the Intent Agent."""

import json
import logging
import math
from copy import deepcopy
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command, interrupt
from pydantic import Field

from agent.agents.stylist.agent import StylistAgent
from agent.agents.stylist.schema import OutfitPlan, StylistRequest
from agent.observability.telemetry import trace_span
from agent.search.categories import (
    category_compatible,
    normalize_category,
    normalize_color,
    wardrobe_display_name,
)
from agent.search.schema import (
    DEFAULT_SEARCH_TOP_K,
    MAX_SEARCH_TOP_K,
    OutfitItemMatch,
    Product,
    ProductConstraints,
    ProductSearchResult,
    SearchRequest,
)
from agent.search.wardrobe import WardrobeInventory, WardrobeSearchService

logger = logging.getLogger(__name__)


def _latest_user_message(state) -> str:
    """Return the current turn's wording for the stylist's language choice."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _tool_message(runtime: ToolRuntime, payload: dict[str, object]) -> list[ToolMessage]:
    logger.info("function_start function=_tool_message")
    return [ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id or "unknown-tool-call",
    )]


def _snapshot(state) -> dict[str, object] | None:
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
        if (previous := old_items.get(item.outfit_item_id)) is not None
        and (previous.category, previous.item_name, previous.color)
        == (item.category, item.item_name, item.color)
    }


def _reusable_matches(
    state: dict[str, object],
    outfit: OutfitPlan,
    unchanged: set[str],
    inventory: WardrobeInventory,
    min_similarity: float = 0.0,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Keep verified results and selections for unchanged, still-owned items."""
    available = {item.item_id: item for item in inventory.items}
    previous_matches = state.get("product_matches", {}) or {}
    previous_selected = state.get("selected_products", {}) or {}
    matches: dict[str, dict[str, object]] = {}
    selected: dict[str, dict[str, object]] = {}
    for item in outfit.items:
        key = item.outfit_item_id
        if key not in unchanged:
            continue
        old_match = previous_matches.get(key, {})
        products = []
        for product in old_match.get("products", []) if isinstance(old_match, dict) else []:
            if not isinstance(product, dict):
                continue
            try:
                similarity = float(product.get("similarity", 0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(similarity) or similarity < min_similarity:
                continue
            candidate = available.get(str(product.get("product_id")))
            if candidate and category_compatible(
                item.category, item.item_name, candidate.category, candidate.name
            ):
                verified = deepcopy(product)
                verified.update({
                    "name": wardrobe_display_name(
                        candidate.category, candidate.color, candidate.name
                    ),
                    "category": normalize_category(candidate.category, candidate.name)
                    or candidate.category,
                    "color": candidate.color,
                })
                products.append(verified)
        chosen = previous_selected.get(key)
        chosen_id = str(chosen.get("product_id")) if isinstance(chosen, dict) else ""
        candidate = available.get(chosen_id)
        if candidate and category_compatible(
            item.category, item.item_name, candidate.category, candidate.name
        ):
            verified = deepcopy(chosen)
            verified.update({
                "name": wardrobe_display_name(candidate.category, candidate.color, candidate.name),
                "category": normalize_category(candidate.category, candidate.name)
                or candidate.category,
                "color": candidate.color,
            })
            selected[key] = verified
            products = [verified] + [
                product for product in products if str(product.get("product_id")) != chosen_id
            ]
        if products:
            matches[key] = {
                "outfit_item_id": key,
                "requested_item": item.model_dump(mode="json"),
                "products": products[:DEFAULT_SEARCH_TOP_K],
            }
    return matches, selected


def build_tools(stylist: StylistAgent, wardrobe_search: WardrobeSearchService) -> list[BaseTool]:
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
        preferences: dict[str, object] | None = None,
        feedback: str = "",
        keep_categories: list[str] | None = None,
        keep_outfit_item_ids: list[str] | None = None,
    ) -> Command:
        """Create or revise an outfit and search processed wardrobe uploads."""
        logger.info("function_start function=consult_stylist")
        state = runtime.state
        current = (
            OutfitPlan.model_validate(state["current_outfit"])
            if state.get("current_outfit") else None
        )
        merged_preferences = deepcopy(state.get("preferences", {}))
        merged_preferences.update(preferences or {})
        user_id = str(state.get("user_id", ""))
        inventory = await wardrobe_search.inventory(user_id)
        if not inventory.items:
            logger.info("stylist_skipped reason=no_completed_wardrobe_items")
            return Command(update={
                "current_outfit": None,
                "product_matches": {},
                "selected_products": {},
                "tool_call_count": 1,
                "messages": _tool_message(runtime, {
                    "outfit": None,
                    "reason": "No completed wardrobe items are available for this user.",
                }),
            })
        request = StylistRequest(
            preferences=merged_preferences,
            user_message=_latest_user_message(state),
            current_outfit=current,
            feedback=feedback,
            keep_categories=keep_categories or [],
            keep_outfit_item_ids=keep_outfit_item_ids or [],
            wardrobe_items=[
                item.model_copy(update={
                    "name": wardrobe_display_name(item.category, item.color, item.name)
                })
                for item in inventory.items
            ],
            selected_wardrobe_items={
                key: str(product.get("product_id"))
                for key, product in (state.get("selected_products", {}) or {}).items()
                if isinstance(product, dict) and product.get("product_id")
            },
        )
        result = await stylist.run(request)
        unchanged = _unchanged_item_ids(current, result)
        matches, selected = _reusable_matches(
            state, result, unchanged, inventory,
            getattr(wardrobe_search, "min_similarity", 0.0),
        )
        search_ids = [
            item.outfit_item_id for item in result.items
            if item.outfit_item_id not in matches
        ]
        if search_ids:
            with trace_span("fashion.wardrobe.search"):
                search_result = await wardrobe_search.search(SearchRequest(
                    outfit=result,
                    user_id=user_id,
                    outfit_item_ids=search_ids,
                    constraints=ProductConstraints(in_stock_only=False),
                ), inventory=inventory)
            matches.update({
                match.outfit_item_id: match.model_dump(mode="json")
                for match in search_result.matches
            })
        logger.info(
            "stylist_inventory_context completed=%d reused=%d searched=%d matches=%d",
            len(inventory.items), len(result.items) - len(search_ids), len(search_ids),
            sum(len(match.get("products", [])) for match in matches.values()),
        )
        history = deepcopy(state.get("outfit_history", []))
        if snapshot := _snapshot(state):
            history.append(snapshot)
        version = int(state.get("outfit_version", 0)) + 1
        feedback_history = deepcopy(state.get("feedback_history", []))
        if feedback:
            feedback_history.append({"feedback": feedback, "outfit_version": version})
        payload = {
            "outfit": result.model_dump(mode="json"),
            "outfit_version": version,
            "product_matches": matches,
        }
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
    async def search_wardrobe(
        runtime: ToolRuntime,
        outfit_item_ids: list[str] | None = None,
        color: str | None = None,
        excluded_item_ids: list[str] | None = None,
        alternative_to_item_id: str | None = None,
        feedback: str = "",
        top_k: Annotated[
            int,
            Field(
                ge=DEFAULT_SEARCH_TOP_K,
                le=MAX_SEARCH_TOP_K,
                description="Return exactly five matches per outfit item",
            ),
        ] = DEFAULT_SEARCH_TOP_K,
    ) -> Command:
        """Search this user's uploads, returning at most five matches per outfit item."""
        logger.info("function_start function=search_wardrobe top_k=%d", top_k)
        if not runtime.state.get("current_outfit"):
            raise ValueError("No current outfit exists; consult the stylist first")
        outfit = OutfitPlan.model_validate(runtime.state["current_outfit"])
        user_id = str(runtime.state.get("user_id", ""))
        targets = outfit_item_ids or []
        unknown = set(targets) - {item.outfit_item_id for item in outfit.items}
        if unknown:
            raise ValueError(f"Unknown outfit item IDs: {sorted(unknown)}")
        request = SearchRequest(
            outfit=outfit,
            user_id=user_id,
            outfit_item_ids=targets,
            top_k=top_k,
            alternative_to_product_id=alternative_to_item_id,
            constraints=ProductConstraints(
                color=color,
                excluded_product_ids=excluded_item_ids or [],
                in_stock_only=False,
            ),
        )
        cached: dict[str, OutfitItemMatch] = {}
        inventory: WardrobeInventory | None = None
        if targets and color is None:
            inventory = await wardrobe_search.inventory(user_id)
            reusable, _ = _reusable_matches(
                runtime.state, outfit,
                {item.outfit_item_id for item in outfit.items}, inventory,
                getattr(wardrobe_search, "min_similarity", 0.0),
            )
            excluded = set(excluded_item_ids or [])
            if alternative_to_item_id:
                excluded.add(alternative_to_item_id)
            for item in outfit.items:
                if item.outfit_item_id not in targets:
                    continue
                chosen = (runtime.state.get("selected_products", {}) or {}).get(
                    item.outfit_item_id, {}
                )
                if chosen.get("product_id"):
                    excluded_for_item = excluded | {str(chosen["product_id"])}
                else:
                    excluded_for_item = excluded
                products = []
                for product in reusable.get(item.outfit_item_id, {}).get("products", []):
                    if str(product.get("product_id")) in excluded_for_item:
                        continue
                    try:
                        products.append(Product.model_validate(product))
                    except (TypeError, ValueError):
                        logger.warning(
                            "wardrobe_cached_match_invalid item_id=%s",
                            item.outfit_item_id,
                        )
                if products:
                    cached[item.outfit_item_id] = OutfitItemMatch(
                        outfit_item_id=item.outfit_item_id,
                        requested_item=item,
                        products=products[:top_k],
                    )
        missing = [key for key in targets if key not in cached]
        if not targets or missing:
            if targets:
                request = request.model_copy(update={"outfit_item_ids": missing})
            with trace_span("fashion.wardrobe.search"):
                searched = await wardrobe_search.search(request, inventory=inventory)
        else:
            searched = ProductSearchResult(outfit_name=outfit.outfit_name, matches=[])
        searched_by_id = {match.outfit_item_id: match for match in searched.matches}
        result = ProductSearchResult(
            outfit_name=outfit.outfit_name,
            matches=[
                cached.get(item.outfit_item_id) or searched_by_id.get(item.outfit_item_id)
                or OutfitItemMatch(
                    outfit_item_id=item.outfit_item_id,
                    requested_item=item,
                    products=[],
                )
                for item in outfit.items
                if not targets or item.outfit_item_id in targets
            ],
        )
        logger.info(
            "wardrobe_alternative_context cached=%d searched=%d",
            len(cached), len(searched.matches),
        )
        matches = deepcopy(runtime.state.get("product_matches", {}))
        for match in result.matches:
            matches[match.outfit_item_id] = match.model_dump(mode="json")
        payload = result.model_dump(mode="json")
        feedback_history = deepcopy(runtime.state.get("feedback_history", []))
        if feedback:
            feedback_history.append({
                "feedback": feedback,
                "outfit_version": runtime.state.get("outfit_version", 0),
                "type": "wardrobe_search",
            })
        count = int(runtime.state.get("tool_call_count", 0)) + 1
        logger.info(
            "tool_complete tool=search_wardrobe item_count=%d result_count=%d "
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
        """Restore the exact previous outfit and its saved wardrobe matches."""
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
    async def select_wardrobe_item(
        outfit_item_id: str,
        item_id: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Select a verified uploaded item already returned for an outfit item."""
        logger.info("function_start function=select_wardrobe_item")
        match = runtime.state.get("product_matches", {}).get(outfit_item_id)
        if not match:
            raise ValueError(f"No wardrobe results exist for outfit item '{outfit_item_id}'")
        product = next(
            (
                item for item in match.get("products", [])
                if str(item.get("product_id")) == item_id
            ),
            None,
        )
        if product is None:
            raise ValueError("The requested item ID is not present in verified search results")
        outfit_data = runtime.state.get("current_outfit")
        if not outfit_data:
            raise ValueError("No current outfit exists for this selection")
        outfit = OutfitPlan.model_validate(outfit_data)
        outfit_item = next(
            (item for item in outfit.items if item.outfit_item_id == outfit_item_id), None
        )
        if outfit_item is None:
            raise ValueError("The requested outfit item no longer exists")
        inventory = await wardrobe_search.inventory(str(runtime.state.get("user_id", "")))
        candidate = next(
            (item for item in inventory.items if item.item_id == item_id), None
        )
        if candidate is None or not category_compatible(
            outfit_item.category, outfit_item.item_name, candidate.category, candidate.name
        ):
            raise ValueError("The requested upload is no longer available for this outfit")
        product = deepcopy(product)
        product.update({
            "name": wardrobe_display_name(candidate.category, candidate.color, candidate.name),
            "category": normalize_category(candidate.category, candidate.name)
            or candidate.category,
            "color": candidate.color,
        })
        selected = deepcopy(runtime.state.get("selected_products", {}))
        if any(
            key != outfit_item_id and str(value.get("product_id")) == item_id
            for key, value in selected.items()
        ):
            raise ValueError("The same wardrobe upload is already selected for another outfit item")
        selected[outfit_item_id] = deepcopy(product)
        payload = {"outfit_item_id": outfit_item_id, "selected_product": product}
        logger.info(
            "tool_complete tool=select_wardrobe_item outfit_item_id=%s", outfit_item_id
        )
        return Command(update={
            "selected_products": selected,
            "tool_call_count": 1,
            "messages": _tool_message(runtime, payload),
        })

    @tool
    async def count_wardrobe(runtime: ToolRuntime) -> Command:
        """Count this user's completed wardrobe items without requiring an outfit."""
        user_id = str(runtime.state.get("user_id", ""))
        with trace_span("fashion.wardrobe.count"):
            count = await wardrobe_search.count_items(user_id)
        logger.info("tool_complete tool=count_wardrobe completed=%d", count)
        return Command(update={
            "tool_call_count": 1,
            "messages": _tool_message(runtime, {"count": count, "status": "completed"}),
        })

    @tool
    async def list_wardrobe(
        runtime: ToolRuntime,
        category: str | None = None,
        color: str | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> Command:
        """List verified completed uploads, optionally filtering type and color."""
        user_id = str(runtime.state.get("user_id", ""))
        inventory = await wardrobe_search.inventory(user_id)
        items = [
            item for item in inventory.items
            if (category is None or category_compatible(
                category, category, item.category, item.name
            ))
            and (color is None or normalize_color(item.color) == normalize_color(color))
        ]
        logger.info(
            "tool_complete tool=list_wardrobe completed=%d filtered=%d returned=%d",
            len(inventory.items), len(items), min(len(items), limit),
        )
        return Command(update={
            "tool_call_count": 1,
            "messages": _tool_message(runtime, {
                "count": len(items),
                "items": [
                    item.model_copy(update={
                        "name": wardrobe_display_name(item.category, item.color, item.name)
                    }).model_dump()
                    for item in items[:limit]
                ],
                "truncated": len(items) > limit,
                "status": "completed",
            }),
        })

    return [
        ask_user,
        consult_stylist,
        search_wardrobe,
        restore_previous_outfit,
        select_wardrobe_item,
        count_wardrobe,
        list_wardrobe,
    ]
