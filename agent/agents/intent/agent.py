"""Factory for the dynamic tool-calling Intent Agent."""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_openai import ChatOpenAI

from agents.intent.prompt import INTENT_PROMPT
from agents.intent.state import FashionState
from agents.intent.tools import build_tools
from agents.stylist.agent import StylistAgent
from fashion_agent.config import Settings, settings
from search.service import ProductSearchService


logger = logging.getLogger(__name__)


def create_intent_agent(
    checkpointer: Any,
    stylist: StylistAgent | None = None,
    product_search: ProductSearchService | None = None,
    config: Settings = settings,
) -> Any:
    model = ChatOpenAI(
        model=config.vllm_model,
        base_url=config.vllm_url,
        api_key=config.vllm_api_key,
        timeout=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
        temperature=0,
    )
    tools = build_tools(
        stylist or StylistAgent(config),
        product_search or ProductSearchService(),
    )

    def tool_error(exc: Exception, request: Any) -> str:
        tool_name = request.tool_call.get("name", "tool")
        logger.warning("tool_failure tool=%s error_type=%s", tool_name, type(exc).__name__)
        return (
            f"{tool_name} failed with {type(exc).__name__}. "
            "Do not claim success or invent results; explain the failure or ask "
            "for corrected input."
        )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=INTENT_PROMPT,
        state_schema=FashionState,
        checkpointer=checkpointer,
        middleware=[ToolErrorMiddleware(on_error=tool_error)],
    )
