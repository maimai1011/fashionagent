"""Independent LLM-backed stylist agent with deterministic preservation checks."""

import json
import logging
import time
from collections.abc import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from agent.agents.chat.config import Settings, settings
from agent.agents.chat.logging import agent_content_logging_enabled
from agent.agents.stylist.prompt import STYLIST_PROMPT
from agent.agents.stylist.schema import OutfitPlan, StylistRequest
from agent.agents.stylist.skill_loader import load_fashion_skill
from agent.observability.metrics import record_stylist
from agent.observability.telemetry import trace_span
from agent.search.categories import category_compatible

logger = logging.getLogger(__name__)


class PreservationError(ValueError):
    """Raised when a stylist revision changes an explicitly locked item."""


class StylistAgent:
    """Callable specialist with its own model invocation and output schema."""

    def __init__(
        self,
        config: Settings = settings,
        runner: Callable[
            [list[dict[str, str]]],
            OutfitPlan | dict[str, object] | Awaitable[OutfitPlan | dict[str, object]],
        ] | None = None,
    ) -> None:
        self.config = config
        self._runner = runner
        self._agent = None

    def _build_agent(self):
        if self._agent is None:
            model = ChatOpenAI(
                model=self.config.vllm_model,
                base_url=self.config.vllm_url,
                api_key=self.config.vllm_api_key,
                timeout=self.config.llm_timeout_seconds,
                max_retries=self.config.llm_max_retries,
                temperature=0.2,
            )
            self._agent = create_agent(
                model=model,
                tools=[],
                system_prompt=STYLIST_PROMPT,
                response_format=ToolStrategy(OutfitPlan),
            )
        return self._agent

    async def run(self, request: StylistRequest) -> OutfitPlan:
        started = time.perf_counter()
        logger.info(
            "stylist_inference_start revision=%s preference_keys=%s feedback_chars=%d "
            "keep_categories=%s keep_item_ids=%s user_message_chars=%d",
            request.current_outfit is not None,
            sorted(request.preferences),
            len(request.feedback),
            request.keep_categories,
            request.keep_outfit_item_ids,
            len(request.user_message),
        )
        payload = request.model_dump(mode="json")
        if agent_content_logging_enabled():
            logger.info(
                "stylist_request payload=%s",
                json.dumps(payload, ensure_ascii=False),
            )
        try:
            skill = load_fashion_skill()
            messages = [{
                "role": "user",
                "content": (
                    "FASHION STYLING SKILL:\n"
                    f"{skill}\n\nREQUEST:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            }]
            if self._runner is not None:
                raw = self._runner(messages)
                if hasattr(raw, "__await__"):
                    raw = await raw
                result = raw if isinstance(raw, OutfitPlan) else OutfitPlan.model_validate(raw)
            else:
                with trace_span("fashion.stylist.model"):
                    response = await self._build_agent().ainvoke(
                        {"messages": messages},
                        config={"recursion_limit": 8},
                    )
                result = OutfitPlan.model_validate(response["structured_response"])
            self._validate_preservation(request, result)
        except Exception as exc:
            latency_seconds = time.perf_counter() - started
            logger.error(
                "stylist_inference_failed error_type=%s latency_ms=%.2f",
                type(exc).__name__,
                latency_seconds * 1000,
            )
            record_stylist("failed", latency_seconds)
            raise
        latency_seconds = time.perf_counter() - started
        logger.info(
            "stylist_inference_complete latency_ms=%.2f items=%d revision=%s "
            "item_categories=%s outfit_item_ids=%s",
            latency_seconds * 1000,
            len(result.items),
            request.current_outfit is not None,
            [item.category for item in result.items],
            [item.outfit_item_id for item in result.items],
        )
        if agent_content_logging_enabled():
            logger.info("stylist_result payload=%s", result.model_dump_json())
        record_stylist("completed", latency_seconds)
        return result

    @staticmethod
    def _validate_preservation(request: StylistRequest, result: OutfitPlan) -> None:
        if request.current_outfit is None:
            return
        old = {item.outfit_item_id: item for item in request.current_outfit.items}
        new = {item.outfit_item_id: item for item in result.items}
        locked_ids = set(request.keep_outfit_item_ids)
        locked_ids.update(
            item.outfit_item_id
            for item in request.current_outfit.items
            if any(category_compatible(
                category, category, item.category, item.item_name
            ) for category in request.keep_categories)
        )
        missing = locked_ids - new.keys()
        changed = {
            item_id for item_id in locked_ids & new.keys()
            if old.get(item_id) != new[item_id]
        }
        if missing or changed:
            raise PreservationError(
                f"stylist changed protected items; missing={sorted(missing)}, "
                f"changed={sorted(changed)}"
            )
