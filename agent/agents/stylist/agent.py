"""Independent LLM-backed stylist agent with deterministic preservation checks."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from agents.stylist.prompt import STYLIST_PROMPT
from agents.stylist.schema import OutfitPlan, StylistRequest
from agents.stylist.skill_loader import load_fashion_skill
from fashion_agent.config import Settings, settings


logger = logging.getLogger(__name__)


class PreservationError(ValueError):
    """Raised when a stylist revision changes an explicitly locked item."""


class StylistAgent:
    """Callable specialist with its own model invocation and output schema."""

    def __init__(
        self,
        config: Settings = settings,
        runner: Callable[[list[dict[str, str]]], Any] | None = None,
    ) -> None:
        logger.info("function_start function=StylistAgent.__init__")
        self.config = config
        self._runner = runner
        self._agent: Any = None

    def _build_agent(self) -> Any:
        logger.info("function_start function=StylistAgent._build_agent")
        if self._agent is None:
            from langchain.agents import create_agent
            from langchain.agents.structured_output import ToolStrategy
            from langchain_openai import ChatOpenAI

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
        logger.info("function_start function=StylistAgent.run")
        started = time.perf_counter()
        skill = load_fashion_skill()
        payload = request.model_dump(mode="json")
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
            response = await self._build_agent().ainvoke(
                {"messages": messages},
                config={"recursion_limit": 8},
            )
            result = OutfitPlan.model_validate(response["structured_response"])
        self._validate_preservation(request, result)
        logger.info(
            "stylist_inference_complete latency_ms=%.2f items=%d revision=%s",
            (time.perf_counter() - started) * 1000,
            len(result.items),
            request.current_outfit is not None,
        )
        return result

    @staticmethod
    def _validate_preservation(request: StylistRequest, result: OutfitPlan) -> None:
        logger.info("function_start function=StylistAgent._validate_preservation")
        if request.current_outfit is None:
            return
        old = {item.outfit_item_id: item for item in request.current_outfit.items}
        new = {item.outfit_item_id: item for item in result.items}
        locked_ids = set(request.keep_outfit_item_ids)
        locked_ids.update(
            item.outfit_item_id
            for item in request.current_outfit.items
            if item.category in set(request.keep_categories)
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
