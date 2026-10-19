"""Factory for the dynamic tool-calling Intent Agent."""

import json
import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolErrorMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.agents.chat.config import Settings, settings
from agent.agents.chat.logging import agent_content_logging_enabled
from agent.agents.intent.prompt import INTENT_PROMPT
from agent.agents.intent.state import FashionState
from agent.agents.intent.tools import build_tools
from agent.agents.stylist.agent import StylistAgent
from agent.observability.metrics import record_intent
from agent.search.wardrobe import WardrobeSearchService

logger = logging.getLogger(__name__)


def _log_intent_model_start(state: AgentState) -> None:
    messages = state.get("messages", [])
    latest_user = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )
    logger.info(
        "intent_model_start messages=%d outfit_version=%s latest_user_chars=%d",
        len(messages),
        state.get("outfit_version", 0),
        len(str(latest_user.content)) if latest_user else 0,
    )
    if agent_content_logging_enabled() and latest_user is not None:
        logger.info(
            "intent_model_input latest_user=%s",
            json.dumps(latest_user.content, ensure_ascii=False, default=str),
        )


def _log_intent_decision(state: AgentState) -> None:
    """Record routing metadata, with full output only when explicitly enabled."""
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        logger.warning("intent_decision outcome=missing_model_response")
        record_intent("missing_model_response")
        return

    response = messages[-1]
    if response.tool_calls:
        calls = [
            {
                "tool": call["name"],
                "arg_keys": sorted(call["args"]),
            }
            for call in response.tool_calls
        ]
        logger.info("intent_decision outcome=tool_calls calls=%s", calls)
        record_intent("tool_calls")
    elif not response.invalid_tool_calls:
        logger.info(
            "intent_decision outcome=direct_reply response_chars=%d",
            len(str(response.content)),
        )
        record_intent("direct_reply")

    if response.invalid_tool_calls:
        logger.warning(
            "intent_decision outcome=invalid_tool_calls count=%d",
            len(response.invalid_tool_calls),
        )
        record_intent("invalid_tool_calls")
    if agent_content_logging_enabled():
        logger.info(
            "intent_model_output content=%s tool_calls=%s invalid_tool_calls=%s",
            json.dumps(response.content, ensure_ascii=False, default=str),
            json.dumps(response.tool_calls, ensure_ascii=False, default=str),
            json.dumps(response.invalid_tool_calls, ensure_ascii=False, default=str),
        )


class IntentDecisionLogger(AgentMiddleware):
    """Log each model decision for synchronous and asynchronous agent runs."""

    def before_model(self, state: AgentState, runtime) -> None:
        _log_intent_model_start(state)

    async def abefore_model(self, state: AgentState, runtime) -> None:
        _log_intent_model_start(state)

    def after_model(self, state: AgentState, runtime) -> None:
        _log_intent_decision(state)

    async def aafter_model(self, state: AgentState, runtime) -> None:
        _log_intent_decision(state)


def create_intent_agent(
    checkpointer: BaseCheckpointSaver,
    stylist: StylistAgent | None = None,
    wardrobe_search: WardrobeSearchService | None = None,
    config: Settings = settings,
):
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
        wardrobe_search or WardrobeSearchService(),
    )
    logger.info(
        "intent_agent_initialized model=%s tools=%s",
        config.vllm_model,
        [tool.name for tool in tools],
    )

    def tool_error(exc: Exception, request) -> str:
        tool_name = request.tool_call.get("name", "tool")
        logger.warning("tool_failure tool=%s error_type=%s", tool_name, type(exc).__name__)
        return (
            f"{tool_name} failed with {type(exc).__name__}. "
            "Do not claim success or invent results. Retry the tool now only if "
            "useful; otherwise explain the failure or ask for corrected input. "
            "Never tell the user to wait for work after your final reply."
        )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=INTENT_PROMPT,
        state_schema=FashionState,
        checkpointer=checkpointer,
        middleware=[IntentDecisionLogger(), ToolErrorMiddleware(on_error=tool_error)],
    )
