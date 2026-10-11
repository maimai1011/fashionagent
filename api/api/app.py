"""FastAPI entrypoint for the fashion assistant."""

import hmac
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.intent.agent import create_intent_agent
from agents.stylist.agent import StylistAgent
from fashion_agent.application import ChatResponse, FashionApplication
from fashion_agent.config import settings
from fashion_agent.logging import configure_logging, request_id_var
from persistence.checkpoint import checkpointer_context
from search.service import ProductSearchService


configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10_000)
    resume: bool = False


def authenticated_user(
    x_user_id: Annotated[str, Header(min_length=1)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    try:
        configured_users = json.loads(os.getenv("APP_USER_TOKENS_JSON") or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Authentication is misconfigured") from exc
    if not isinstance(configured_users, dict):
        raise HTTPException(status_code=500, detail="Authentication is misconfigured")
    expected = configured_users.get(x_user_id) if configured_users else os.getenv("APP_API_KEY")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not isinstance(expected, str) or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid API credentials")
    return x_user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    product_search = ProductSearchService()
    try:
        if os.getenv("VALIDATE_MILVUS_ON_STARTUP", "true").lower() == "true":
            await product_search.validate()
        async with checkpointer_context() as checkpointer:
            graph = create_intent_agent(
                checkpointer,
                stylist=StylistAgent(),
                product_search=product_search,
            )
            app.state.fashion = FashionApplication(graph)
            yield
    finally:
        product_search.close()


app = FastAPI(title="Fashion Multi-Agent API", version="0.1.0", lifespan=lifespan)
_request_times: defaultdict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    identity = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _request_times[identity]
    while window and window[0] <= now - 60:
        window.popleft()
    if len(window) >= settings.api_rate_limit_per_minute:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    window.append(now)
    return await call_next(request)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request_token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        logger.info(
            "http_request request_id=%s method=%s path=%s latency_ms=%.2f",
            request_id, request.method, request.url.path,
            (time.perf_counter() - started) * 1000,
        )
        request_id_var.reset(request_token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    user_id: Annotated[str, Depends(authenticated_user)],
) -> ChatResponse:
    try:
        return await request.app.state.fashion.chat(
            user_id=user_id,
            thread_id=body.thread_id,
            message=body.message,
            resume=body.resume,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "fashion_request_failed user_id=%s thread_id=%s", user_id, body.thread_id
        )
        raise HTTPException(
            status_code=503,
            detail="Fashion service is temporarily unavailable",
        ) from exc
