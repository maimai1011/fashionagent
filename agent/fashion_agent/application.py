"""Session-safe application facade over the checkpointed agent graph."""

import asyncio
import logging
import time
from typing import Any
from weakref import WeakValueDictionary

from langgraph.types import Command
from pydantic import BaseModel, Field

from fashion_agent.config import Settings, settings
from fashion_agent.logging import session_id_var


logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    thread_id: str
    message: str = ""
    interrupted: bool = False
    clarification: str | None = None
    outfit: dict[str, Any] | None = None
    product_matches: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selected_products: dict[str, dict[str, Any]] = Field(default_factory=dict)
    outfit_version: int = 0


class FashionApplication:
    def __init__(self, agent: Any, config: Settings = settings) -> None:
        self.agent = agent
        self.config = config
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @staticmethod
    def _session_key(user_id: str, thread_id: str) -> str:
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
        if not message.strip():
            raise ValueError("message cannot be empty")
        session = self._session_key(user_id, thread_id)
        run_config = {
            "configurable": {"thread_id": session},
            "recursion_limit": self.config.recursion_limit,
        }
        lock = self._locks.get(session)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session] = lock
        async with lock:
            session_token = session_id_var.set(session)
            started = time.perf_counter()
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
                payload: Any
                if resume:
                    payload = Command(resume=message)
                else:
                    payload = {
                        "messages": [{"role": "user", "content": message}],
                        "user_id": user_id,
                    }
                async with asyncio.timeout(self.config.request_timeout_seconds):
                    result = await self.agent.ainvoke(payload, config=run_config)
                response = self._serialize(thread_id, result)
                logger.info(
                    "fashion_turn_complete latency_ms=%.2f interrupted=%s",
                    (time.perf_counter() - started) * 1000,
                    response.interrupted,
                )
                return response
            finally:
                session_id_var.reset(session_token)

    @staticmethod
    def _serialize(thread_id: str, result: dict[str, Any]) -> ChatResponse:
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
        return ChatResponse(
            thread_id=thread_id,
            message=message,
            outfit=result.get("current_outfit"),
            product_matches=result.get("product_matches", {}),
            selected_products=result.get("selected_products", {}),
            outfit_version=result.get("outfit_version", 0),
        )
