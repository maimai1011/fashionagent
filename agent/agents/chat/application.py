"""Session-safe application facade over the checkpointed agent graph."""

import asyncio
import logging
import time
from weakref import WeakValueDictionary

from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.agents.chat.config import Settings, settings
from agent.agents.chat.logging import request_id_var, session_id_var
from agent.observability.metrics import record_chat
from agent.observability.telemetry import trace_span

logger = logging.getLogger(__name__)


def _deferred_reply(message: str) -> bool:
    """Catch final answers that promise work after this synchronous request ends."""
    reply = message.casefold()
    return any(phrase in reply for phrase in (
        "hãy đợi", "đợi một chút", "chờ một chút",
        "please wait", "one moment", "hang on",
        "i'll retry", "i will retry", "i'll continue searching",
        "i will continue searching",
    ))


class ChatResponse(BaseModel):
    thread_id: str
    message: str = ""
    interrupted: bool = False
    clarification: str | None = None
    outfit: dict[str, object] | None = None
    product_matches: dict[str, dict[str, object]] = Field(default_factory=dict)
    selected_products: dict[str, dict[str, object]] = Field(default_factory=dict)
    outfit_version: int = 0


class FashionApplication:
    def __init__(self, agent, config: Settings = settings) -> None:
        logger.info("function_start function=FashionApplication.__init__")
        self.agent = agent
        self.config = config
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @staticmethod
    def _session_key(user_id: str, thread_id: str) -> str:
        logger.info("function_start function=FashionApplication._session_key")
        if not user_id.strip() or not thread_id.strip():
            raise ValueError("user_id and thread_id are required")
        return f"{user_id.strip()}::{thread_id.strip()}"

    async def chat(
        self,
        user_id: str,
        thread_id: str,
        message: str,
        *,
        resume: bool = False,
    ) -> ChatResponse:
        logger.info("function_start function=FashionApplication.chat")
        if not message.strip():
            raise ValueError("message cannot be empty")
        session = self._session_key(user_id, thread_id)
        run_config = {
            "configurable": {"thread_id": session},
            "recursion_limit": self.config.recursion_limit,
            "metadata": {
                "request_id": request_id_var.get(),
                "session_id": session,
            },
        }
        lock = self._locks.get(session)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session] = lock
        async with lock:
            session_token = session_id_var.set(session)
            started = time.perf_counter()
            outcome = "failed"
            try:
                if hasattr(self.agent, "aget_state"):
                    snapshot = await self.agent.aget_state(run_config)
                    pending = any(
                        getattr(task, "interrupts", ())
                        for task in getattr(snapshot, "tasks", ())
                    )
                    if resume and not pending:
                        raise ValueError("No clarification is waiting for a resume answer")
                    if pending and not resume:
                        raise ValueError(
                            "A clarification is pending; submit the answer with resume=true"
                        )
                if resume:
                    payload = Command(resume=message)
                else:
                    payload = {
                        "messages": [{"role": "user", "content": message}],
                        "user_id": user_id,
                    }
                async with asyncio.timeout(self.config.request_timeout_seconds):
                    with trace_span("fashion.chat.turn"):
                        result = await self.agent.ainvoke(payload, config=run_config)
                response = self._serialize(thread_id, result)
                logger.info(
                    "fashion_turn_complete latency_ms=%.2f interrupted=%s",
                    (time.perf_counter() - started) * 1000,
                    response.interrupted,
                )
                outcome = "interrupted" if response.interrupted else "completed"
                return response
            finally:
                record_chat(outcome, time.perf_counter() - started)
                session_id_var.reset(session_token)

    @staticmethod
    def _serialize(thread_id: str, result: dict[str, object]) -> ChatResponse:
        logger.info("function_start function=FashionApplication._serialize")
        interrupts = result.get("__interrupt__", ())
        if interrupts:
            value = interrupts[0].value
            question = value.get("question") if isinstance(value, dict) else str(value)
            return ChatResponse(
                thread_id=thread_id,
                interrupted=True,
                clarification=question,
                outfit=result.get("current_outfit"),
                product_matches=result.get("product_matches", {}),
                selected_products=result.get("selected_products", {}),
                outfit_version=result.get("outfit_version", 0),
            )
        message = ""
        messages = result.get("messages", [])
        if messages:
            content = getattr(messages[-1], "content", "")
            if isinstance(content, str):
                message = content
            elif isinstance(content, list):
                message = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
        if _deferred_reply(message):
            logger.warning("agent_deferred_reply_replaced")
            message = (
                "Tôi chưa hoàn tất tìm kiếm trong lượt này. "
                "Vui lòng thử lại; nếu lỗi tiếp tục, hãy kiểm tra dịch vụ tìm kiếm."
                if any(phrase in message.casefold() for phrase in (
                    "hãy đợi", "đợi một chút", "chờ một chút"
                ))
                else "I couldn't finish the search in this turn. Please try again; "
                "if it keeps failing, check the search service."
            )
        return ChatResponse(
            thread_id=thread_id,
            message=message,
            outfit=result.get("current_outfit"),
            product_matches=result.get("product_matches", {}),
            selected_products=result.get("selected_products", {}),
            outfit_version=result.get("outfit_version", 0),
        )
